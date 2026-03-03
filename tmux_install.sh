#!/usr/bin/env bash
set -euo pipefail

# Combined installer:
# 1) Ensures Node.js >= 20, installs Claude/Codex CLIs
# 2) Builds and installs tmux from source tarball via curl

CLAUDE_NPM_PACKAGE="${CLAUDE_NPM_PACKAGE:-@anthropic-ai/claude-code}"
CODEX_NPM_PACKAGE="${CODEX_NPM_PACKAGE:-@openai/codex}"
MIN_NODE_MAJOR="${MIN_NODE_MAJOR:-20}"
PREFERRED_NODE_MAJOR="${PREFERRED_NODE_MAJOR:-22}"
TMUX_VERSION="${TMUX_VERSION:-3.5a}"

if [[ "${EUID}" -eq 0 ]]; then
  PREFIX="${PREFIX:-/usr/local}"
else
  PREFIX="${PREFIX:-$HOME/.local}"
fi

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

node_major() {
  node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0
}

persist_user_local_bin_path() {
  if [[ ! -f "$HOME/.bashrc" ]]; then
    touch "$HOME/.bashrc"
  fi
  if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  fi
  export PATH="$HOME/.local/bin:$PATH"
}

persist_node22_path() {
  local marker npm_global_bin
  marker="# codex-installer-node-path"
  if [[ ! -f "$HOME/.bashrc" ]]; then
    touch "$HOME/.bashrc"
  fi
  npm_global_bin="$(npm bin -g 2>/dev/null || true)"
  if ! grep -Fq "${marker}" "$HOME/.bashrc"; then
    cat >> "$HOME/.bashrc" <<EOF
# codex-installer-node-path
for d in "\$HOME"/.local/node-v22.*/bin; do
  if [ -d "\$d" ]; then
    export PATH="\$d:\$PATH"
    break
  fi
done
if [ -d "${npm_global_bin}" ]; then
  export PATH="${npm_global_bin}:\$PATH"
fi
EOF
  fi
}

install_node_user_space() {
  local os arch ver tarball url dir
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "${arch}" in
    x86_64) arch="x64" ;;
    aarch64) arch="arm64" ;;
    *)
      echo "Unsupported architecture for auto Node install: ${arch}" >&2
      return 1
      ;;
  esac

  ver="$(curl -fsSL "https://nodejs.org/dist/latest-v${PREFERRED_NODE_MAJOR}.x/SHASUMS256.txt" | awk 'NR==1{gsub(/node-v|-.*/,"",$2); print $2}')"
  if [[ -z "${ver}" ]]; then
    echo "Unable to resolve latest Node ${PREFERRED_NODE_MAJOR}.x version." >&2
    return 1
  fi

  tarball="node-v${ver}-${os}-${arch}.tar.xz"
  url="https://nodejs.org/dist/v${ver}/${tarball}"
  dir="$HOME/.local/node-v${ver}-${os}-${arch}"

  mkdir -p "$HOME/.local"
  curl -fsSL "${url}" -o "/tmp/${tarball}"
  tar -xJf "/tmp/${tarball}" -C "$HOME/.local"
  rm -f "/tmp/${tarball}"

  export PATH="${dir}/bin:${PATH}"
}

install_node_and_npm_if_missing() {
  if need_cmd node && need_cmd npm; then
    return 0
  fi
  if need_cmd apt-get; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs npm
    return 0
  fi
  if need_cmd apk; then
    apk add --no-cache nodejs npm
    return 0
  fi
  if need_cmd dnf; then
    dnf install -y nodejs npm
    return 0
  fi
  if need_cmd yum; then
    yum install -y nodejs npm
    return 0
  fi
}

ensure_node_compatible() {
  install_node_and_npm_if_missing || true
  if need_cmd node && need_cmd npm && [[ "$(node_major)" -ge "${MIN_NODE_MAJOR}" ]]; then
    return 0
  fi
  echo "Node.js >= ${MIN_NODE_MAJOR} is required. Installing Node ${PREFERRED_NODE_MAJOR}.x in user space..."
  install_node_user_space
  if ! need_cmd node || ! need_cmd npm || [[ "$(node_major)" -lt "${MIN_NODE_MAJOR}" ]]; then
    echo "Failed to provision compatible Node.js runtime." >&2
    exit 1
  fi
}

install_claude_codex() {
  ensure_node_compatible
  persist_node22_path
  export PATH="$(dirname "$(command -v node)"):${PATH}"
  local npm_global_bin
  npm_global_bin="$(npm bin -g 2>/dev/null || true)"
  if [[ -n "${npm_global_bin}" ]]; then
    export PATH="${npm_global_bin}:${PATH}"
  fi

  echo "Installing Claude package: ${CLAUDE_NPM_PACKAGE}"
  npm install -g "${CLAUDE_NPM_PACKAGE}"
  echo "Installing Codex package: ${CODEX_NPM_PACKAGE}"
  npm install -g "${CODEX_NPM_PACKAGE}"
}

have_tmux_build_libs() {
  if ! need_cmd pkg-config; then
    return 1
  fi
  pkg-config --exists libevent_core libevent_extra ncurses || pkg-config --exists libevent ncurses
}

install_tmux_build_deps_if_possible() {
  if need_cmd apt-get; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      curl ca-certificates build-essential pkg-config bison automake autoconf \
      libevent-dev libncurses-dev
    return 0
  fi
  if need_cmd apk; then
    apk add --no-cache \
      curl ca-certificates build-base pkgconf bison automake autoconf \
      libevent-dev ncurses-dev
    return 0
  fi
  if need_cmd dnf; then
    dnf install -y \
      curl ca-certificates gcc make pkgconf-pkg-config bison automake autoconf \
      libevent-devel ncurses-devel
    return 0
  fi
  if need_cmd yum; then
    yum install -y \
      curl ca-certificates gcc make pkgconfig bison automake autoconf \
      libevent-devel ncurses-devel
    return 0
  fi
}

print_tmux_dep_help() {
  cat >&2 <<'EOF'
Missing tmux build dependencies (libevent and/or ncurses development files).
Debian/Ubuntu:
  apt-get update && apt-get install -y libevent-dev libncurses-dev build-essential pkg-config bison automake autoconf
Alpine:
  apk add --no-cache libevent-dev ncurses-dev build-base pkgconf bison automake autoconf
RHEL/CentOS/Fedora:
  yum install -y libevent-devel ncurses-devel gcc make pkgconfig bison automake autoconf
  # or
  dnf install -y libevent-devel ncurses-devel gcc make pkgconf-pkg-config bison automake autoconf
EOF
}

install_tmux_with_curl() {
  local tmpdir tarball url
  install_tmux_build_deps_if_possible || true

  for cmd in curl make gcc tar; do
    if ! need_cmd "${cmd}"; then
      echo "Missing required command for tmux build: ${cmd}" >&2
      exit 1
    fi
  done

  if ! have_tmux_build_libs; then
    print_tmux_dep_help
    exit 1
  fi

  mkdir -p "${PREFIX}"
  tmpdir="$(mktemp -d)"
  # Expand tmpdir now to avoid unbound-variable errors with set -u on RETURN trap.
  trap "rm -rf '${tmpdir}'" RETURN

  tarball="tmux-${TMUX_VERSION}.tar.gz"
  url="https://github.com/tmux/tmux/releases/download/${TMUX_VERSION}/${tarball}"
  cd "${tmpdir}"
  echo "Downloading ${url}"
  curl -fsSL -o "${tarball}" "${url}"
  tar -xzf "${tarball}"
  cd "tmux-${TMUX_VERSION}"
  echo "Building tmux ${TMUX_VERSION}..."
  ./configure --prefix="${PREFIX}"
  make -j"$(nproc)"
  make install

  if [[ "${PREFIX}" == "$HOME/.local" ]]; then
    persist_user_local_bin_path
  fi
}

main() {
  install_claude_codex
  install_tmux_with_curl

  echo
  echo "Install complete. Versions:"
  if need_cmd node; then node -v || true; fi
  if need_cmd claude; then claude --version || true; fi
  if need_cmd codex; then codex --version || true; fi
  if [[ -x "${PREFIX}/bin/tmux" ]]; then
    "${PREFIX}/bin/tmux" -V || true
  elif need_cmd tmux; then
    tmux -V || true
  fi
}

main "$@"
