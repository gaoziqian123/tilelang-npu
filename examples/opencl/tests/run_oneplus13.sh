#!/usr/bin/env bash
# Emit OpenCL examples, build/deploy tl_probe, and verify on OnePlus 13.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENCL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
TILELANG_ROOT=$(cd "$OPENCL_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$TILELANG_ROOT/.." && pwd)
PROBE_DIR="$PROJECT_ROOT/backend/gpu/tl_probe"
PY=${PYTHON:-"$TILELANG_ROOT/.venv/bin/python"}
PHONE_HOST=${OPENCL_SSH_HOST:-oneplus13-reverse}
REMOTE_DIR=${OPENCL_REMOTE_DIR:-'$HOME/tl_opencl'}
SSH_OPT=(-o ControlPath="$HOME/.ssh/cm-%C")

export PYTHONPATH=${PYTHONPATH:-$TILELANG_ROOT}

if ! ssh "${SSH_OPT[@]}" -o ConnectTimeout=8 "$PHONE_HOST" true 2>/dev/null; then
  for _ in $(seq 1 15); do
    ssh -o ControlMaster=yes -o ControlPath="$HOME/.ssh/cm-%C" \
        -o ControlPersist=30m -o ConnectTimeout=10 -fN "$PHONE_HOST" 2>/dev/null && break
    sleep 6
  done
fi

"$PY" "$OPENCL_DIR/silu/silu.py"
"$PY" "$OPENCL_DIR/rmsnorm/rmsnorm.py"
"$PY" "$OPENCL_DIR/gemm/gemm_nt.py"

ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-/root/autodl-tmp/android-ndk-r28b} "$PROBE_DIR/build_android.sh"

ssh "${SSH_OPT[@]}" "$PHONE_HOST" "mkdir -p $REMOTE_DIR"
scp "${SSH_OPT[@]}" \
  "$PROBE_DIR/build/tl_probe" \
  "$OPENCL_DIR/silu/out/silu.cl" \
  "$OPENCL_DIR/rmsnorm/out/rmsnorm.cl" \
  "$OPENCL_DIR/gemm/out/gemm_nt.cl" \
  "$PHONE_HOST:$REMOTE_DIR/"

local_md5=$(
  tmp=$(mktemp -d)
  cp "$PROBE_DIR/build/tl_probe" "$OPENCL_DIR/silu/out/silu.cl" \
     "$OPENCL_DIR/rmsnorm/out/rmsnorm.cl" "$OPENCL_DIR/gemm/out/gemm_nt.cl" "$tmp/"
  (cd "$tmp" && md5sum tl_probe silu.cl rmsnorm.cl gemm_nt.cl)
  rm -rf "$tmp"
)
remote_md5=$(ssh "${SSH_OPT[@]}" "$PHONE_HOST" "cd $REMOTE_DIR && md5sum tl_probe silu.cl rmsnorm.cl gemm_nt.cl" | awk '{print $1"  "$2}')
if [[ "$local_md5" != "$remote_md5" ]]; then
  echo "MD5 mismatch" >&2
  printf 'local:\n%s\nremote:\n%s\n' "$local_md5" "$remote_md5" >&2
  exit 2
fi
echo "MD5_OK"

tag="tl_opencl_$(date +%Y%m%d_%H%M%S)"
ssh "${SSH_OPT[@]}" "$PHONE_HOST" "cd $REMOTE_DIR && rm -f $tag.log $tag.done && nohup sh -c 'LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 ./tl_probe silu.cl silu_kernel_kernel silu; r1=\$?; LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 ./tl_probe rmsnorm.cl rmsnorm_kernel_kernel rmsnorm; r2=\$?; LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 ./tl_probe gemm_nt.cl gemm_nt_kernel_kernel gemm; r3=\$?; rc=0; [ \$r1 -eq 0 ] || rc=\$r1; [ \$r2 -eq 0 ] || rc=\$r2; [ \$r3 -eq 0 ] || rc=\$r3; echo \$rc > $tag.done; exit \$rc' > $tag.log 2>&1 &"

for _ in $(seq 1 120); do
  if ssh "${SSH_OPT[@]}" "$PHONE_HOST" "test -f $REMOTE_DIR/$tag.done"; then
    break
  fi
  sleep 5
done

ssh "${SSH_OPT[@]}" "$PHONE_HOST" "cat $REMOTE_DIR/$tag.log"
rc=$(ssh "${SSH_OPT[@]}" "$PHONE_HOST" "cat $REMOTE_DIR/$tag.done 2>/dev/null || echo 124")
echo "REMOTE_RC $rc"
exit "$rc"
