# Baked pod image: ComfyUI + VideoHelperSuite + kohya sd-scripts venv + all deps pre-installed on the
# image's local layers (fast). Models + LoRA outputs + datasets stay on the RunPod network volume at
# /workspace at runtime (the volume shadows /workspace, so NOTHING app-related is baked there — code
# lives at /opt). Boot = pull image (~3min) + start; no pip/venv on the slow network volume, ever.
#
# Built + pushed by .github/workflows/build.yml → ghcr.io/zeropercentsam/mg-pod:latest (make the GHCR
# package PUBLIC so RunPod can pull without creds). To use: COMFY_IMAGE=ghcr.io/zeropercentsam/mg-pod:latest
# and UNSET COMFY_BOOT_URL (createPod then uses this image's CMD instead of curling boot.sh).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg wget procps && rm -rf /var/lib/apt/lists/*

# ComfyUI (deps into the base python = base CUDA torch 2.4.1) at /opt (NOT /workspace — volume shadows it)
RUN git clone --depth=1 https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
 && python3 -m pip install -r /opt/ComfyUI/requirements.txt

# VideoHelperSuite (VHS_VideoCombine → mp4) + imageio-ffmpeg for the encoder
RUN git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite \
 && python3 -m pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt imageio-ffmpeg

# kohya sd-scripts in a CLEAN, self-contained venv with pinned cu124 torch (consistent transformers →
# fixes the "Could not import CLIPTextModel" that the system-site-packages hybrid caused on the volume).
RUN git clone --depth=1 https://github.com/kohya-ss/sd-scripts /opt/sd-scripts \
 && python3 -m venv /opt/kohya-venv \
 && /opt/kohya-venv/bin/pip install --upgrade pip \
 && /opt/kohya-venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 \
 && cd /opt/sd-scripts && /opt/kohya-venv/bin/pip install -r requirements.txt accelerate \
 && /opt/kohya-venv/bin/python -c "import torch; from transformers import CLIPTextModel; print('kohya env OK')"

COPY authproxy.py /opt/authproxy.py
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

EXPOSE 8188
CMD ["/opt/entrypoint.sh"]
