#!/usr/bin/env bash
# No-image pod bootstrap. Runs via RunPod dockerStartCmd on a STOCK pytorch/cuda image:
#   bash -lc "curl -fsSL $COMFY_BOOT_URL | bash"
# Brings ComfyUI up on 127.0.0.1:8189 behind a tiny Python auth-proxy on :8188 that requires
# `Authorization: Bearer $COMFYUI_TOKEN`. Pure stdlib proxy (no caddy download to stall on).
# The proxy also serves a /train endpoint (LoRA training via kohya sd-scripts) — see below.
# Secrets come from pod env (COMFYUI_TOKEN, HF_TOKEN, model URLs) — none baked in.
# NOTE: the proxy forwards HTTP only (fire-and-poll). ComfyUI WS progress is not proxied (the app
# polls /history, so this is fine); add a WS-capable proxy later if live progress is wanted.
# Not using `set -e`: the proxy must come up even if a provisioning step fails, so the pod is
# always reachable + gated and failures are observable via the API.

: "${COMFYUI_TOKEN:?COMFYUI_TOKEN required}"
WS=/workspace
CD="$WS/ComfyUI"

echo "[boot] tools"
command -v git >/dev/null 2>&1 || (apt-get update -y && apt-get install -y --no-install-recommends git) >/dev/null 2>&1 || true

# --- ComfyUI on the volume (persists across restarts) ---
if [ ! -d "$CD" ]; then
  echo "[boot] cloning ComfyUI"
  git clone --depth=1 https://github.com/comfyanonymous/ComfyUI "$CD" && pip install -q -r "$CD/requirements.txt" || echo "[boot] WARN ComfyUI install issue"
fi

# --- base models: download whenever a URL is set (idempotent; needed for generation even in MINIMAL) ---
M="$CD/models"
mkdir -p "$M"/{checkpoints,vae,loras,controlnet,upscale_models,diffusion_models,text_encoders,clip_vision,instantid,pulid}
dl() {
  [ -z "${1:-}" ] && return 0
  [ -f "$2" ] && return 0
  echo "[boot] downloading $(basename "$2")"
  if [ -n "${HF_TOKEN:-}" ]; then wget -q --header="Authorization: Bearer ${HF_TOKEN}" -O "$2" "$1" || { echo "[boot] WARN dl failed $2"; rm -f "$2"; }
  else wget -q -O "$2" "$1" || { echo "[boot] WARN dl failed $2"; rm -f "$2"; }; fi
}
dl "${SDXL_CKPT_URL:-}"    "$M/checkpoints/sdxl-base.safetensors"
dl "${FLUX_DEV_FP8_URL:-}" "$M/diffusion_models/flux1-dev-fp8.safetensors"
dl "${FLUX_VAE_URL:-}"     "$M/vae/ae.safetensors"
dl "${T5XXL_URL:-}"        "$M/text_encoders/t5xxl_fp16.safetensors"
dl "${CLIP_L_URL:-}"       "$M/text_encoders/clip_l.safetensors"
dl "${LTXV_CKPT_URL:-}"    "$M/checkpoints/ltx-video.safetensors"   # Phase 5 img2vid (needs T5XXL_URL too)

# --- custom nodes: only on a full boot (MINIMAL=1 skips; heavy clones + pip for Phase 4/5) ---
if [ "${MINIMAL:-0}" != "1" ] && [ ! -f "$WS/.provisioned" ]; then
  echo "[boot] provisioning custom nodes (first full boot — may take 5–20 min)"
  CN="$CD/custom_nodes"; mkdir -p "$CN"
  clone() { d="$CN/$(basename "$1")"; [ -d "$d" ] || git clone --depth=1 "$1" "$d"; [ -f "$d/requirements.txt" ] && pip install -q -r "$d/requirements.txt" || true; }
  clone https://github.com/ltdrdata/ComfyUI-Manager
  clone https://github.com/ltdrdata/ComfyUI-Impact-Pack
  clone https://github.com/cubiq/ComfyUI_IPAdapter_plus
  clone https://github.com/cubiq/ComfyUI-InstantID
  clone https://github.com/Gourieff/comfyui-reactor-node
  clone https://github.com/cubiq/PuLID_ComfyUI
  clone https://github.com/Fannovel16/comfyui_controlnet_aux
  clone https://github.com/kijai/ComfyUI-FluxTrainer
  clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale
  clone https://github.com/kijai/ComfyUI-WanVideoWrapper
  clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite   # VHS_VideoCombine → mp4 (Phase 5)
  touch "$WS/.provisioned"
fi

# --- kohya sd-scripts for SDXL LoRA training (full boot only; venv isolates kohya's pinned deps
#     from ComfyUI's env so neither breaks the other). Persists on the volume → one-time cost. ---
if [ "${MINIMAL:-0}" != "1" ] && [ ! -f "$WS/.trainer_provisioned" ]; then
  echo "[boot] provisioning kohya sd-scripts (LoRA trainer — may take a few min)"
  [ -d "$WS/sd-scripts" ] || git clone --depth=1 https://github.com/kohya-ss/sd-scripts "$WS/sd-scripts"
  # --system-site-packages reuses the base CUDA torch (kohya reqs don't pin torch) → avoids a 2GB reinstall.
  python -m venv --system-site-packages "$WS/kohya-venv"
  "$WS/kohya-venv/bin/pip" install -q --upgrade pip
  "$WS/kohya-venv/bin/pip" install -q -r "$WS/sd-scripts/requirements.txt" || echo "[boot] WARN kohya reqs"
  "$WS/kohya-venv/bin/pip" install -q accelerate bitsandbytes xformers || echo "[boot] WARN kohya extras"
  touch "$WS/.trainer_provisioned"
fi

# --- auth-proxy: :8188 (Bearer-gated) → ComfyUI :8189, plus a local /train endpoint (kohya) ---
cat > /authproxy.py <<'PY'
import os, re, json, time, threading, subprocess, shutil, traceback, urllib.request, urllib.error, http.server
from urllib.parse import urlparse, parse_qs

TOKEN = os.environ.get("COMFYUI_TOKEN", "")
UP = "http://127.0.0.1:8189"
WS = "/workspace"
LORA_DIR = f"{WS}/ComfyUI/models/loras"
SDXL = f"{WS}/ComfyUI/models/checkpoints/sdxl-base.safetensors"
KOHYA = f"{WS}/sd-scripts"
ACCEL = f"{WS}/kohya-venv/bin/accelerate"
JOBS_FILE = f"{WS}/train/jobs.json"
HOP = {"host", "authorization", "content-length", "connection", "transfer-encoding"}

# In-memory job table, mirrored to JOBS_FILE so a pod stop/start doesn't lose a long train's status.
# ponytail: one GPU → one job at a time (new job rejected with 409 while one runs). Add a real queue
# only if concurrent training is ever wanted.
jobs = {}
jobs_lock = threading.Lock()

def _load_jobs():
    global jobs
    try:
        with open(JOBS_FILE) as f: jobs = json.load(f)
    except Exception: jobs = {}

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
    req = urllib.request.Request(url, headers={"User-Agent": "mg-pod"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)

def _run_training(job_id, spec):
    try:
        token = (spec.get("instance_token") or "ohwx woman").strip()
        out_name = spec["output_name"]
        steps = int(spec.get("steps", 1200))
        dim = int(spec.get("network_dim", 32))
        alpha = int(spec.get("network_alpha", 16))
        repeats = int(spec.get("repeats", 10))
        images = spec["images"]
        captions = spec.get("captions") or []
        # token is free-text from the app's trigger-word field → sanitize before it becomes a path component
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
            return _set(job_id, status="error", error=f"SDXL base not on volume ({SDXL}); set SDXL_CKPT_URL + full-boot once")
        _set(job_id, status="training", progress=5)
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
            "--optimizer_type=AdamW8bit", "--lr_scheduler=cosine",
            "--mixed_precision=fp16", "--save_precision=fp16",
            "--cache_latents", "--gradient_checkpointing",
            "--save_model_as=safetensors", "--caption_extension=.txt",
            "--seed=42", "--no_half_vae", "--xformers",
        ]
        with open(log_path, "w") as lf:
            p = subprocess.run(cmd, cwd=KOHYA, stdout=lf, stderr=subprocess.STDOUT)
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
        return self._proxy("GET")
    def do_POST(self):
        if not self._auth(): return self._send(401, b"unauthorized")
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
        with jobs_lock: st = jobs.get(job_id)
        if not st: return self._json(404, {"status": "unknown", "error": "unknown job (pod may have been recreated)"})
        return self._json(200, st)
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
PY

echo "[boot] starting ComfyUI (127.0.0.1:8189)"
cd "$CD" && python main.py --listen 127.0.0.1 --port 8189 >/comfy.log 2>&1 &
for _ in $(seq 1 150); do curl -sf "http://127.0.0.1:8189/system_stats" >/dev/null 2>&1 && break; sleep 2; done
echo "[boot] starting auth-proxy on :8188"
exec python3 /authproxy.py
