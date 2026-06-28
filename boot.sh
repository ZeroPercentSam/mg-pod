#!/usr/bin/env bash
# No-image pod bootstrap. Runs via RunPod dockerStartCmd on a STOCK pytorch/cuda image:
#   bash -lc "curl -fsSL $COMFY_BOOT_URL | bash"
# Brings ComfyUI up on 127.0.0.1:8189 behind a tiny Python auth-proxy on :8188 that requires
# `Authorization: Bearer $COMFYUI_TOKEN`. Pure stdlib proxy (no caddy download to stall on).
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

# --- one-time provisioning: custom nodes + base models (MINIMAL=1 skips it for fast boot) ---
if [ "${MINIMAL:-0}" != "1" ] && [ ! -f "$WS/.provisioned" ]; then
  echo "[boot] provisioning custom nodes + models (first boot — may take 5–30 min)"
  CN="$CD/custom_nodes"; M="$CD/models"
  mkdir -p "$CN" "$M"/{checkpoints,vae,loras,controlnet,upscale_models,diffusion_models,text_encoders,clip_vision,instantid,pulid}
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
  dl() { [ -z "${1:-}" ] && return 0; [ -f "$2" ] && return 0; echo "[boot] dl $(basename "$2")"; wget -q --header="Authorization: Bearer ${HF_TOKEN:-}" -O "$2" "$1" || echo "[boot] WARN download failed: $2"; }
  dl "${SDXL_CKPT_URL:-}"    "$M/checkpoints/sdxl-base.safetensors"
  dl "${FLUX_DEV_FP8_URL:-}" "$M/diffusion_models/flux1-dev-fp8.safetensors"
  dl "${FLUX_VAE_URL:-}"     "$M/vae/ae.safetensors"
  dl "${T5XXL_URL:-}"        "$M/text_encoders/t5xxl_fp16.safetensors"
  dl "${CLIP_L_URL:-}"       "$M/text_encoders/clip_l.safetensors"
  touch "$WS/.provisioned"
fi

# --- tiny stdlib auth-proxy: :8188 (Bearer-gated) → 127.0.0.1:8189 ---
cat > /authproxy.py <<'PY'
import os, http.server, urllib.request, urllib.error
TOKEN = os.environ.get("COMFYUI_TOKEN", "")
UP = "http://127.0.0.1:8189"
HOP = {"host", "authorization", "content-length", "connection", "transfer-encoding"}
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def _send(self, code, data=b"", ctype=None):
        self.send_response(code)
        if ctype: self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data: self.wfile.write(data)
    def _proxy(self, method):
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            return self._send(401, b"unauthorized")
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
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            d = e.read(); self._send(e.code, d)
        except Exception as e:
            self._send(502, str(e).encode())
    def do_GET(self): self._proxy("GET")
    def do_POST(self): self._proxy("POST")
    def do_DELETE(self): self._proxy("DELETE")
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("0.0.0.0", 8188), H).serve_forever()
PY

echo "[boot] starting ComfyUI (127.0.0.1:8189)"
cd "$CD" && python main.py --listen 127.0.0.1 --port 8189 >/comfy.log 2>&1 &
for _ in $(seq 1 150); do curl -sf "http://127.0.0.1:8189/system_stats" >/dev/null 2>&1 && break; sleep 2; done
echo "[boot] starting auth-proxy on :8188"
exec python3 /authproxy.py
