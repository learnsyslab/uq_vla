#!/usr/bin/env bash
# Install the LIBERO-Plus benchmark next to hf-libero, for the LIBERO-Plus failure-detection rollouts
# (scripts/failure_detection/run_libero_plus.sh). Everything goes to third_party/libero_plus/:
#   LIBERO-plus/   the LIBERO-Plus fork at a fixed commit, plus its 9.5 GB of assets
#   config/        a LIBERO config.yaml pointing at the fork
#   shim/          put FIRST on PYTHONPATH: makes `import libero` resolve to the fork instead of the
#                  installed hf-libero (the fork is a namespace package and cannot be pip-installed)
#   env.sh         exports LIBERO_PLUS_SHIM and LIBERO_CONFIG_PATH; sourced by the rollout script
# Only the LIBERO-Plus recordings use the shim; everything else keeps using hf-libero.
#
#   bash scripts/setup_libero_plus.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD/third_party/libero_plus"
FORK="$ROOT/LIBERO-plus"
COMMIT=4976dc30028e805ff8094b55501d532c48fec182
mkdir -p "$ROOT"

if [ ! -d "$FORK/.git" ]; then
    git clone https://github.com/sylvestf/LIBERO-plus "$FORK"
fi
git -C "$FORK" checkout -q "$COMMIT"
# Optional speed/portability patch of the perturbation wrapper: ImageMagick becomes optional (cv2
# fallback for motion blur) and the glass-blur pixel shuffle is numba-compiled (~80x faster).
if ! git -C "$FORK" diff --quiet -- libero/libero/envs/env_wrapper.py; then
    echo "env_wrapper.py already patched"
else
    git -C "$FORK" apply "$PWD/third_party/libero_plus_env_wrapper.patch"
fi

# Assets (6.4 GB download, 9.5 GB unpacked).
LIB="$FORK/libero/libero"
if [ ! -e "$LIB/assets" ]; then
    huggingface-cli download Sylvest/LIBERO-plus assets.zip --repo-type dataset --local-dir "$ROOT/download"
    unzip -q "$ROOT/download/assets.zip" -d "$LIB"
    ln -s "$(find "$LIB" -type d -path '*LIBERO-plus-0/assets' | head -1)" "$LIB/assets"
    rm -f "$ROOT/download/assets.zip"
fi

mkdir -p "$ROOT/config"
cat > "$ROOT/config/config.yaml" <<YAML
benchmark_root: $LIB
bddl_files: $LIB/bddl_files
init_states: $LIB/init_files
datasets: $FORK/libero/datasets
assets: $LIB/assets
YAML

mkdir -p "$ROOT/shim/libero"
: > "$ROOT/shim/libero/__init__.py"   # a regular package shadows the installed namespace package
ln -sfn "$LIB" "$ROOT/shim/libero/libero"

cat > "$ROOT/env.sh" <<ENV
export LIBERO_PLUS_SHIM="$ROOT/shim"
export LIBERO_CONFIG_PATH="$ROOT/config"
ENV
echo "LIBERO-Plus ready in $ROOT"
