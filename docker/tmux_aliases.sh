#!/usr/bin/env bash
# Short tmux aliases for pod use
# Source this file or add to ~/.bashrc:
#   source /path/to/tmux_aliases.sh

# tn [name]   - new session (optional name)
tn() { tmux new-session ${1:+-s "$1"}; }

# ta [name]   - attach to session (latest if no name given)
ta() { tmux attach-session ${1:+-t "$1"}; }

# tl          - list sessions
alias tl='tmux list-sessions'

# tk [name]   - kill session (current if no name given)
tk() { tmux kill-session ${1:+-t "$1"}; }

# ts <name>   - switch to named session
ts() { tmux switch-client -t "$1"; }

# tw [name]   - new window (optional name)
tw() { tmux new-window ${1:+-n "$1"}; }

# trn <name>  - rename current session
trn() { tmux rename-session "$1"; }

# tgpu [interval_secs]  - open a new window showing GPU utilization (default: every 60s)
tgpu() {
  local interval="${1:-60}"
  tmux new-window -n gpu "watch -n ${interval} nvidia-smi"
}
