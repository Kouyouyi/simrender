#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-sam2}"

install_sam2() {
    local python_bin="${SAM2_PYTHON:-python}"
    echo "Installing bundled SAM2 runtime with ${python_bin}"
    SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}" \
        "${python_bin}" -m pip install --no-build-isolation \
        -e "${repo_root}/third_party/sam2"
}

install_propainter() {
    local python_bin="${PROPAINTER_PYTHON:-python}"
    echo "Installing bundled ProPainter dependencies with ${python_bin}"
    "${python_bin}" -m pip install \
        -r "${repo_root}/third_party/ProPainter/requirements.txt"
}

case "${target}" in
    sam2)
        install_sam2
        ;;
    propainter)
        install_propainter
        ;;
    all)
        install_sam2
        install_propainter
        ;;
    *)
        echo "Usage: $0 [sam2|propainter|all]" >&2
        exit 2
        ;;
esac

cat <<EOF
Runtime sources:
  SAM2:       ${repo_root}/third_party/sam2
  ProPainter: ${repo_root}/third_party/ProPainter

Download model weights next:
  ${repo_root}/scripts/download_egoview_checkpoints.sh
EOF
