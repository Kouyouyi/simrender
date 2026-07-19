# Third-party runtime sources

This directory vendors the weight-free runtime portions required by the Galbot
hand-removal pipeline. Model checkpoints, generated frames, videos, caches, and
datasets are not committed.

| Project | Upstream | Pinned commit | Included content | License |
|---|---|---|---|---|
| SAM 2 | https://github.com/facebookresearch/sam2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Python package, configs, build metadata, checkpoint downloader, upstream docs | Apache-2.0; see `sam2/LICENSE` |
| ProPainter | https://github.com/sczhou/ProPainter | `e870e79321c31b733e2031af5aa2fb1fe3ac7eec` | Inference entrypoint and its RAFT/model/core/utils runtime dependencies | S-Lab License 1.0; see `ProPainter/LICENSE` |

SAM 2 examples, notebooks, demo applications, training data, and checkpoints
are excluded. ProPainter examples, test media, results, training datasets, and
checkpoints are excluded.

**ProPainter is licensed for non-commercial redistribution and use only.** For
commercial use, obtain permission from its contributors as described in the
upstream license. The Apache-2.0 license of simrender does not override either
third-party license.

Use `scripts/setup_egoview.sh` to install dependencies and
`scripts/download_egoview_checkpoints.sh` to obtain the unversioned weights.
