#!/usr/bin/env bash
# No-image pod bootstrap. Runs via RunPod dockerStartCmd on a STOCK pytorch/cuda image:
#   bash -lc "curl -fsSL $COMFY_BOOT_URL | bash"
# Brings ComfyUI up on 127.0.0.1:8189 behind a tiny Python auth-proxy on :8188 that requires
# `Authorization: Bearer $COMFYUI_TOKEN`. Pure stdlib proxy (no caddy download to stall on).
# The proxy also serves /train (LoRA training via kohya sd-scripts) + /boot-log (provisioning trace).
# Secrets come from pod env (COMFYUI_TOKEN, HF_TOKEN, model URLs) — none baked in.
#
# The auth-proxy starts FIRST, then provisioning + ComfyUI run after it — so /boot-log and /train/status
# are reachable from t=0 and a long (20–40min) first boot is observable instead of a blind black box.
# ComfyUI :8189 isn't up until provisioning finishes, so /system_stats 502s until then (expected).
# python3 explicitly everywhere: this image ships python3 but bare `python` is not on PATH in the boot
# context (using `python` silently broke venv creation + ComfyUI start on earlier boots).
# Not using `set -e`: the proxy must stay up even if a provisioning step fails, so the pod is always
# reachable + gated and failures are observable via the API.

: "${COMFYUI_TOKEN:?COMFYUI_TOKEN required}"
WS=/workspace
CD="$WS/ComfyUI"
mkdir -p "$WS"
exec > >(tee -a "$WS/boot.log") 2>&1   # persist boot trace to the volume → readable live via /boot-log
# Warm-boot fast path: every provisioning step below is guarded by a sentinel on the volume, so a 2nd+
# boot skips straight to starting ComfyUI (~2min). Set REPROVISION=1 to force a clean re-provision
# (e.g. after bumping ComfyUI/nodes) without wiping the whole volume.
[ "${REPROVISION:-}" = "1" ] && { echo "[boot] REPROVISION=1 → clearing sentinels"; rm -f "$WS"/.deps-comfy "$WS"/.deps-vhs "$WS"/.kohya-v2 "$WS"/.provisioned; }

# --- auth-proxy: :8188 (Bearer-gated) → ComfyUI :8189, plus local /train + /boot-log endpoints ---
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
    # No worker thread survives a pod restart, so any job still marked in-flight is orphaned. Fail it,
    # else the 409 busy-check would reject every future /train forever (and the app would poll it forever).
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
    # http(s) only — block file:// (local-file read) and other schemes from the job spec.
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"refusing non-http(s) dataset url: {url[:48]}")
    req = urllib.request.Request(url, headers={"User-Agent": "mg-pod"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)

def _run_training(job_id, spec):
    try:
        token = (spec.get("instance_token") or "ohwx woman").strip()
        # output_name becomes a file path + kohya --output_name → strip anything that could escape LORA_DIR
        out_name = re.sub(r"[^A-Za-z0-9_-]", "", spec.get("output_name", ""))
        if not out_name:
            return _set(job_id, status="error", error="invalid or missing output_name")
        # `or default` (not get(k, default)) so an explicit null from the app doesn't reach int(None)
        steps = int(spec.get("steps") or 1200)
        dim = int(spec.get("network_dim") or 32)
        alpha = int(spec.get("network_alpha") or 16)
        repeats = int(spec.get("repeats") or 10)
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
            "--optimizer_type=AdamW", "--lr_scheduler=cosine",
            "--mixed_precision=fp16", "--save_precision=fp16",
            "--cache_latents", "--gradient_checkpointing",
            "--save_model_as=safetensors", "--caption_extension=.txt",
            "--seed=42", "--no_half_vae", "--sdpa",
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
        if self.path.startswith("/boot-log"): return self._boot_log()
        return self._proxy("GET")
    def do_POST(self):
        if not self._auth():
            n = int(self.headers.get("Content-Length") or 0)
            if n: self.rfile.read(n)  # drain body so the HTTP/1.1 keep-alive connection stays in sync
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
            return self._json(503, {"error": "trainer not provisioned yet (still booting, or boot with COMFY_MINIMAL unset)"})
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
        # While training, kohya's tqdm writes "<pct>%" to the log — surface the latest so the bar moves
        # instead of sitting at the single value set when the run started.
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
PY
python3 /authproxy.py &
echo "[boot] auth-proxy up on :8188 (provisioning below — watch GET /boot-log)"

echo "[boot] tools"
command -v git >/dev/null 2>&1 || (apt-get update -y && apt-get install -y --no-install-recommends git) >/dev/null 2>&1 || true

# --- ComfyUI on the volume (persists across restarts) ---
if [ ! -d "$CD" ]; then
  echo "[boot] cloning ComfyUI"
  git clone --depth=1 https://github.com/comfyanonymous/ComfyUI "$CD" || echo "[boot] WARN ComfyUI clone"
fi
if [ ! -f "$WS/.deps-comfy" ]; then
  echo "[boot] installing ComfyUI deps"
  python3 -m pip install -q -r "$CD/requirements.txt" && touch "$WS/.deps-comfy" || echo "[boot] WARN ComfyUI deps (retry next boot)"
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

CN="$CD/custom_nodes"; mkdir -p "$CN"
# --- VideoHelperSuite: the ONLY custom node a built-in template needs (VHS_VideoCombine → mp4 for
#     ltx-img2vid). Clone + ENSURE deps EVERY boot (python3) — a prior boot's broken bare-pip can leave
#     node deps uninstalled, and the optional-node flag below would otherwise skip reinstall. mp4 uses
#     imageio-ffmpeg (pulled by VHS requirements) — no apt ffmpeg (apt update can hang the boot). ---
VHS="$CN/ComfyUI-VideoHelperSuite"
[ -d "$VHS" ] || git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite "$VHS"
if [ ! -f "$WS/.deps-vhs" ]; then
  echo "[boot] installing VideoHelperSuite deps"
  python3 -m pip install -q -r "$VHS/requirements.txt" && python3 -m pip install -q imageio-ffmpeg && touch "$WS/.deps-vhs" || echo "[boot] WARN VHS deps (retry next boot)"
fi

# --- optional heavy nodes (face-swap / controlnet / upscale / etc.): first full boot only. Not used by
#     any current template, so their import failures are harmless noise — kept for future work. ---
if [ "${MINIMAL:-0}" != "1" ] && [ ! -f "$WS/.provisioned" ]; then
  echo "[boot] provisioning optional custom nodes (first full boot — may take 5–20 min)"
  clone() { d="$CN/$(basename "$1")"; [ -d "$d" ] || git clone --depth=1 "$1" "$d"; [ -f "$d/requirements.txt" ] && python3 -m pip install -q -r "$d/requirements.txt" || true; }
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
  touch "$WS/.provisioned"
fi

# --- kohya sd-scripts for SDXL LoRA training (full boot only). CLEAN self-contained venv (NOT
#     --system-site-packages): the system-site hybrid made the venv load the base transformers against a
#     mismatched torch → "Could not import CLIPTextModel". Clean venv + pinned cu124 torch fixes it.
#     Sentinel .kohya-v2 forces a one-time rebuild over any earlier broken venv. ---
if [ "${MINIMAL:-0}" != "1" ] && [ ! -f "$WS/.kohya-v2" ]; then
  echo "[boot] provisioning kohya sd-scripts (clean venv v2 — downloads torch, a few min)"
  [ -d "$WS/sd-scripts" ] || git clone --depth=1 https://github.com/kohya-ss/sd-scripts "$WS/sd-scripts"
  rm -rf "$WS/kohya-venv"
  python3 -m venv "$WS/kohya-venv" || echo "[boot] WARN venv create failed"
  PIP="$WS/kohya-venv/bin/pip"
  "$PIP" install --upgrade pip
  "$PIP" install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 || echo "[boot] WARN torch"
  # cd so the trailing `.` in requirements.txt (installs the local kohya `library` pkg) resolves
  (cd "$WS/sd-scripts" && "$PIP" install -r requirements.txt) || echo "[boot] WARN kohya reqs"
  "$PIP" install accelerate || echo "[boot] WARN accelerate"
  # Smoke-test the exact import that failed before; only flag provisioned if it passes (else retry).
  if "$WS/kohya-venv/bin/python" -c "import torch; from transformers import CLIPTextModel" 2>>"$WS/boot.log"; then
    echo "[boot] kohya env OK"; touch "$WS/.kohya-v2"
  else
    echo "[boot] WARN kohya env import FAILED — will retry next boot"
  fi
fi

echo "[boot] starting ComfyUI (127.0.0.1:8189)"
cd "$CD" && python3 main.py --listen 127.0.0.1 --port 8189 >/comfy.log 2>&1 &
echo "[boot] provisioning complete; ComfyUI starting. Pod ready once /system_stats returns 200."
wait   # keep the container alive on the auth-proxy (+ ComfyUI) background jobs
