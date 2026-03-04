# Inference image for round-trip-code-compression zero-shot runs.
# Built on the official vLLM image so CUDA, vLLM, and cuDNN are pre-installed.
# Usage:
#   docker build -t nickil21/round-trip-code-compression:latest .
#   docker push nickil21/round-trip-code-compression:latest

FROM vllm/vllm-openai:v0.9.2

WORKDIR /workspace

ENV HF_HOME=/mnt_path/huggingface \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# Install tmux via apt (fast). tmux_install.sh will skip its source build if tmux is found.
RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*

# Install Node, Claude Code, and Codex CLIs.
# Done before COPY . . so this layer is cached independently of source changes.
COPY tmux_install.sh .
RUN bash tmux_install.sh

# Install short tmux aliases (tn/ta/tl/tk/ts/tw/trn).
COPY tmux_aliases.sh .
RUN echo 'source /workspace/tmux_aliases.sh' >> ~/.bashrc

# tmux_install.sh writes PATH updates to ~/.bashrc (login-shell only).
# Symlink claude and codex into /usr/local/bin so they are reachable in
# non-interactive, non-login shells (e.g. the container entrypoint).
RUN bash -lc '\
    for cmd in claude codex; do \
      p=$(command -v "$cmd" 2>/dev/null || true); \
      if [ -n "$p" ] && [ ! -e "/usr/local/bin/$cmd" ]; then \
        ln -sf "$p" /usr/local/bin/"$cmd"; \
      fi; \
    done'

# Install project dependencies.
# vllm is already provided by the base image and is intentionally commented out
# in requirements.txt, so this installs only the remaining packages.
COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed blinker -r requirements.txt

# Copy project source and make scripts executable.
COPY . .
RUN chmod +x scripts/*.sh
