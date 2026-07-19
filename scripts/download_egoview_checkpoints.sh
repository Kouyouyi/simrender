#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sam_dir="${SAM2_MODEL_DIR:-${repo_root}/checkpoints/sam2}"
propainter_dir="${PROPAINTER_MODEL_DIR:-${repo_root}/third_party/ProPainter/weights}"

mkdir -p "${sam_dir}" "${propainter_dir}"

download() {
    local url="$1"
    local destination="$2"
    if [[ -s "${destination}" ]]; then
        echo "Already present: ${destination}"
        return
    fi

    local partial="${destination}.part"
    echo "Downloading ${url}"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --continue-at - \
            --output "${partial}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget --continue --output-document="${partial}" "${url}"
    else
        echo "curl or wget is required" >&2
        exit 1
    fi
    mv "${partial}" "${destination}"
}

sam_base="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
propainter_base="https://github.com/sczhou/ProPainter/releases/download/v0.1.0"

download \
    "${sam_base}/sam2.1_hiera_base_plus.pt" \
    "${sam_dir}/sam2.1_hiera_base_plus.pt"
download "${propainter_base}/raft-things.pth" "${propainter_dir}/raft-things.pth"
download \
    "${propainter_base}/recurrent_flow_completion.pth" \
    "${propainter_dir}/recurrent_flow_completion.pth"
download "${propainter_base}/ProPainter.pth" "${propainter_dir}/ProPainter.pth"

cat <<EOF
Weights are ready and remain ignored by Git.
SAM2_CHECKPOINT=${sam_dir}/sam2.1_hiera_base_plus.pt
PROPAINTER_ROOT=${repo_root}/third_party/ProPainter
EOF
