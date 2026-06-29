#!/usr/bin/env bash
# Baked-image runtime entrypoint. No provisioning — ComfyUI + nodes + kohya venv are already in the image
# at /opt. Only models + LoRA outputs + datasets live on the network volume (/workspace), so this just:
# point ComfyUI's model dir at the volume, download base models if missing (one-time), start proxy + ComfyUI.
# Not `set -e`: the proxy must come up even if a model download fails (so the pod is reachable + observable).
: "${COMFYUI_TOKEN:?COMFYUI_TOKEN required}"
WS=/workspace
CD=/opt/ComfyUI
MODELS="$WS/ComfyUI/models"   # on the volume → models + trained LoRAs persist across pods
mkdir -p "$WS"
exec > >(tee -a "$WS/boot.log") 2>&1
echo "[boot] baked image entrypoint"

# Point ComfyUI's (baked, empty) model tree at the volume so checkpoints/loras persist + are shared.
mkdir -p "$MODELS"/{checkpoints,vae,loras,controlnet,upscale_models,diffusion_models,text_encoders,clip_vision}
rm -rf "$CD/models"
ln -sfn "$MODELS" "$CD/models"

# Download base models to the volume if missing (big sequential files — fine over the volume; one-time).
dl() {
  [ -z "${1:-}" ] && return 0
  [ -f "$2" ] && return 0
  echo "[boot] downloading $(basename "$2")"
  if [ -n "${HF_TOKEN:-}" ]; then wget -q --header="Authorization: Bearer ${HF_TOKEN}" -O "$2" "$1" || { echo "[boot] WARN dl $2"; rm -f "$2"; }
  else wget -q -O "$2" "$1" || { echo "[boot] WARN dl $2"; rm -f "$2"; }; fi
}
dl "${SDXL_CKPT_URL:-}" "$MODELS/checkpoints/sdxl-base.safetensors"
dl "${T5XXL_URL:-}"     "$MODELS/text_encoders/t5xxl_fp16.safetensors"
dl "${CLIP_L_URL:-}"    "$MODELS/text_encoders/clip_l.safetensors"
dl "${LTXV_CKPT_URL:-}" "$MODELS/checkpoints/ltx-video.safetensors"

python3 /opt/authproxy.py &
echo "[boot] auth-proxy up on :8188"
cd "$CD" && python3 main.py --listen 127.0.0.1 --port 8189 >/comfy.log 2>&1 &
echo "[boot] ComfyUI starting; ready once /system_stats returns 200"
wait
