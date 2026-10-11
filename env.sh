export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$PROJECT_ROOT/.cache/hf"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
export WANDB_DIR="$PROJECT_ROOT/outputs/wandb"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR" "$WANDB_DIR"
[ -f "$PROJECT_ROOT/.venv/bin/activate" ] && source "$PROJECT_ROOT/.venv/bin/activate" || true
