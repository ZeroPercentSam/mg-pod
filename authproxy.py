#!/usr/bin/env python3
# Bearer-gated proxy (:8188) → ComfyUI (:8189), plus local /train (kohya LoRA) + /boot-log endpoints.
# Standalone version for the BAKED Docker image (mg-pod): code + kohya venv live on the image at /opt
# (fast local layer); models + LoRA outputs + datasets live on the network volume at /workspace.
# Kept in sync with the inline copy in boot.sh (the no-Docker fallback). Pure stdlib — no deps.
import os, re, json, time, threading, subprocess, shutil, traceback, urllib.request, urllib.error, http.server
from urllib.parse import urlparse, parse_qs

TOKEN = os.environ.get("COMFYUI_TOKEN", "")
UP = "http://127.0.0.1:8189"
WS = "/workspace"
LORA_DIR = f"{WS}/ComfyUI/models/loras"
SDXL = f"{WS}/ComfyUI/models/checkpoints/sdxl-base.safetensors"
KOHYA = "/opt/sd-scripts"
ACCEL = "/opt/kohya-venv/bin/accelerate"
JOBS_FILE = f"{WS}/train/jobs.json"
HOP = {"host", "authorization", "content-length", "connection", "transfer-encoding"}

jobs = {}
jobs_lock = threading.Lock()

def _load_jobs():
    global jobs
    try:
        with open(JOBS_FILE) as f: jobs = json.load(f)
    except Exception: jobs = {}
    # No worker thread survives a pod restart — fail any orphaned in-flight job so the busy-check frees up.
    changed = False
    for v in jobs.values():
        if v.get("status") in ("queued", "downloading", "training"):
            v["status"], v["error"] = "error", "pod restarted mid-training"
            changed = True
    if changed:
        _save_jobs()

def _save_jobs():
    try:
        os.makedirs(os.path.dirname(JOBS_FILE), exist_ok=True)
        tmp = JOBS_FILE + ".tmp"
        with open(tmp, "w") as f: json.dump(jobs, f)
        os.replace(tmp, JOBS_FILE)
    except Exception: pass

def _set(job_id, **kw):
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(kw)
        _save_jobs()

def _download(url, dest):
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"refusing non-http(s) dataset url: {url[:48]}")
    req = urllib.request.Request(url, headers={"User-Agent": "mg-pod"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)

# ComfyUI + kohya share ONE GPU. ComfyUI caches model weights in VRAM after gens (and /free leaves
# a large CUDA-context residual), which OOMs the trainer. So we fully stop ComfyUI for the training's
# duration to hand it the whole card, then bring it back. PID1 (entrypoint) waits only on the proxy,
# so killing ComfyUI here does NOT end the container.
COMFY_ARGS = ["python3", "main.py", "--listen", "127.0.0.1", "--port", "8189"]
def _comfy_up():
    try:
        urllib.request.urlopen(UP + "/system_stats", timeout=5).read(); return True
    except Exception:
        return False
def _stop_comfy():
    subprocess.run(["pkill", "-9", "-f", "main.py --listen 127.0.0.1 --port 8189"])
    for _ in range(20):
        if not _comfy_up(): return
        time.sleep(1)
def _start_comfy():
    subprocess.Popen(COMFY_ARGS, cwd="/opt/ComfyUI", stdout=open("/comfy.log", "a"), stderr=subprocess.STDOUT)
    for _ in range(60):  # ~2min for ComfyUI to reload + be reachable again
        if _comfy_up(): return True
        time.sleep(2)
    return False

def _run_training(job_id, spec):
    try:
        token = (spec.get("instance_token") or "ohwx woman").strip()
        out_name = re.sub(r"[^A-Za-z0-9_-]", "", spec.get("output_name", ""))
        if not out_name:
            return _set(job_id, status="error", error="invalid or missing output_name")
        steps = int(spec.get("steps") or 1200)
        dim = int(spec.get("network_dim") or 32)
        alpha = int(spec.get("network_alpha") or 16)
        repeats = int(spec.get("repeats") or 10)
        images = spec["images"]
        captions = spec.get("captions") or []
        token_dir = re.sub(r"[^A-Za-z0-9 _-]", "", token).strip() or "sks"
        root = f"{WS}/train/{job_id}"
        img_dir = f"{root}/img/{repeats}_{token_dir}"
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(LORA_DIR, exist_ok=True)
        _set(job_id, status="downloading", progress=2, count=len(images))
        for i, url in enumerate(images):
            ext = ".png"
            for e in (".png", ".jpg", ".jpeg", ".webp"):
                if url.lower().split("?")[0].endswith(e): ext = e; break
            base = f"{img_dir}/img_{i:03d}"
            _download(url, base + ext)
            cap = captions[i] if i < len(captions) and captions[i] else token
            with open(base + ".txt", "w") as f: f.write(cap)
        if not os.path.exists(SDXL):
            return _set(job_id, status="error", error=f"SDXL base not on volume ({SDXL}); set SDXL_CKPT_URL")
        _set(job_id, status="training", progress=5)
        _stop_comfy()  # free the whole GPU for kohya (shared card); ComfyUI restarts after, see finally
        log_path = f"{root}/train.log"
        cmd = [
            ACCEL, "launch", "--num_processes=1", "--num_machines=1",
            "--mixed_precision=fp16", "--dynamo_backend=no",
            f"{KOHYA}/sdxl_train_network.py",
            f"--pretrained_model_name_or_path={SDXL}",
            f"--train_data_dir={root}/img",
            f"--output_dir={LORA_DIR}",
            f"--output_name={out_name}",
            "--resolution=1024,1024",
            "--network_module=networks.lora",
            f"--network_dim={dim}", f"--network_alpha={alpha}",
            "--train_batch_size=1",
            f"--max_train_steps={steps}",
            "--learning_rate=1e-4", "--unet_lr=1e-4", "--text_encoder_lr=5e-5",
            "--optimizer_type=AdamW", "--lr_scheduler=cosine",
            "--mixed_precision=fp16", "--save_precision=fp16",
            "--cache_latents", "--gradient_checkpointing",
            "--save_model_as=safetensors", "--caption_extension=.txt",
            "--seed=42", "--no_half_vae", "--sdpa",
        ]
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # less fragmentation
        try:
            with open(log_path, "w") as lf:
                p = subprocess.run(cmd, cwd=KOHYA, stdout=lf, stderr=subprocess.STDOUT, env=env)
        finally:
            _start_comfy()  # bring ComfyUI back so generations work again post-training
        out_file = f"{LORA_DIR}/{out_name}.safetensors"
        if p.returncode == 0 and os.path.exists(out_file):
            _set(job_id, status="done", progress=100, output=f"{out_name}.safetensors", size=os.path.getsize(out_file))
        else:
            tail = ""
            try:
                with open(log_path) as lf: tail = "".join(lf.readlines()[-40:])
            except Exception: pass
            _set(job_id, status="error", error=f"kohya exit {p.returncode}\n{tail[-2000:]}")
    except Exception as e:
        _set(job_id, status="error", error=f"{e}\n{traceback.format_exc()[-1200:]}")

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def _auth(self): return self.headers.get("Authorization") == f"Bearer {TOKEN}"
    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def _send(self, code, data=b"", ctype=None):
        self.send_response(code)
        if ctype: self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data))); self.end_headers()
        if data: self.wfile.write(data)
    def do_GET(self):
        if not self._auth(): return self._send(401, b"unauthorized")
        if self.path.startswith("/train/status"): return self._train_status()
        if self.path.startswith("/boot-log"): return self._boot_log()
        return self._proxy("GET")
    def do_POST(self):
        if not self._auth():
            n = int(self.headers.get("Content-Length") or 0)
            if n: self.rfile.read(n)
            return self._send(401, b"unauthorized")
        if self.path == "/train": return self._start_train()
        return self._proxy("POST")
    def do_DELETE(self):
        if not self._auth(): return self._send(401, b"unauthorized")
        return self._proxy("DELETE")
    def _start_train(self):
        n = int(self.headers.get("Content-Length") or 0)
        try: spec = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e: return self._json(400, {"error": f"bad json: {e}"})
        if not spec.get("images") or not spec.get("output_name"):
            return self._json(400, {"error": "images[] and output_name required"})
        if not os.path.exists(ACCEL):
            return self._json(503, {"error": "trainer not provisioned on this pod"})
        job_id = spec.get("job_id") or f"job-{int(time.time())}"
        with jobs_lock:
            busy = [j for j, v in jobs.items() if v.get("status") in ("queued", "downloading", "training")]
            if busy: return self._json(409, {"error": f"a training job is already running: {busy[0]}"})
            jobs[job_id] = {"status": "queued", "progress": 0, "error": None, "output": None}
            _save_jobs()
        threading.Thread(target=_run_training, args=(job_id, spec), daemon=True).start()
        return self._json(200, {"job_id": job_id, "status": "queued"})
    def _train_status(self):
        job_id = (parse_qs(urlparse(self.path).query).get("id") or [""])[0]
        with jobs_lock: st = dict(jobs.get(job_id) or {})
        if not st: return self._json(404, {"status": "unknown", "error": "unknown job (pod may have been recreated)"})
        if st.get("status") == "training":
            try:
                with open(f"{WS}/train/{job_id}/train.log", "rb") as f:
                    f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 4096))
                    tail = f.read().decode("utf-8", "ignore")
                pcts = re.findall(r"(\d{1,3})%", tail)
                if pcts: st["progress"] = max(5, min(99, int(pcts[-1])))
            except Exception: pass
        return self._json(200, st)
    def _boot_log(self):
        out = {}
        for name, path in (("boot", f"{WS}/boot.log"), ("comfy", "/comfy.log")):
            try:
                with open(path, "rb") as f:
                    f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 8000))
                    out[name] = f.read().decode("utf-8", "ignore")
            except Exception as e:
                out[name] = f"(no {path}: {e})"
        return self._json(200, out)

    def _proxy(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        req = urllib.request.Request(UP + self.path, data=body, method=method)
        for k, v in self.headers.items():
            if k.lower() not in HOP: req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = r.read()
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() not in HOP: self.send_header(k, v)
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        except urllib.error.HTTPError as e:
            d = e.read(); self._send(e.code, d)
        except Exception as e:
            self._send(502, str(e).encode())
    def log_message(self, *a): pass

_load_jobs()
http.server.ThreadingHTTPServer(("0.0.0.0", 8188), H).serve_forever()
