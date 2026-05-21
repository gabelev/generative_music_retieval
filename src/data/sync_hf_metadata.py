"""Sync small project artifacts (splits CSVs, SemID CSVs, RQ-VAE codebooks,
per-run JSON reports, and the canonical RESULTS.md doc) between local disk
and a private HF dataset repo.

Local paths and HF paths are kept identical (so `data/splits/foo.csv` on disk
mirrors `data/splits/foo.csv` in the repo). The existing audio at `audio/*.wav`
is untouched — it has its own crawl/staging flow.

Modes:
  --mode upload    : push --paths from local to HF
  --mode download  : pull --paths from HF to local

Each --paths entry may be a directory (synced with --patterns filters) or a
single file (synced as-is, no pattern matching).

Auth: HF_TOKEN env var, or `huggingface-cli login`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

DEFAULT_PATHS = [
    "data/splits",
    "data/semantic_ids",
    "runs",
    "docs/RESULTS.md",
]
DEFAULT_PATTERNS = ["*.csv", "*.pkl", "*.json", "*.md"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", required=True,
                        help="e.g. gabelev/discogs-vi-csi-subset")
    parser.add_argument("--mode", required=True, choices=["upload", "download"])
    parser.add_argument("--paths", nargs="+", default=DEFAULT_PATHS,
                        help="Files or directories (paths are reused verbatim on HF).")
    parser.add_argument("--patterns", nargs="+", default=DEFAULT_PATTERNS,
                        help="Patterns applied to directory entries in --paths.")
    args = parser.parse_args()

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or (Path.home() / ".cache" / "huggingface" / "token").exists()):
        sys.exit("HF auth missing: set HF_TOKEN or run `huggingface-cli login`.")

    api = HfApi()

    if args.mode == "upload":
        for path in args.paths:
            p = Path(path)
            if not p.exists():
                print(f"Skip {path} (not present locally)")
                continue
            if p.is_dir():
                matches = []
                for pat in args.patterns:
                    matches.extend(p.glob(pat))
                if not matches:
                    print(f"Skip {path} (no files matching {args.patterns})")
                    continue
                print(f"Uploading {len(matches)} files from {path} -> {args.hf_repo}/{path}/")
                try:
                    api.upload_folder(
                        folder_path=str(p),
                        path_in_repo=path,
                        repo_id=args.hf_repo,
                        repo_type="dataset",
                        allow_patterns=args.patterns,
                        commit_message=f"sync {path} from local",
                    )
                except HfHubHTTPError as e:
                    sys.exit(f"upload_folder failed for {path}: {e}")
            else:
                print(f"Uploading file {path} -> {args.hf_repo}/{path}")
                try:
                    api.upload_file(
                        path_or_fileobj=str(p),
                        path_in_repo=path,
                        repo_id=args.hf_repo,
                        repo_type="dataset",
                        commit_message=f"sync {path} from local",
                    )
                except HfHubHTTPError as e:
                    sys.exit(f"upload_file failed for {path}: {e}")
        print(f"Uploaded. View: https://huggingface.co/datasets/{args.hf_repo}")

    elif args.mode == "download":
        allow_patterns: list[str] = []
        for path in args.paths:
            # If it looks like a file (has a recognized extension), include as-is.
            # Otherwise treat as a directory and apply patterns.
            if "." in Path(path).name and "/" not in path[-30:].split(".")[-1]:
                allow_patterns.append(path)
            else:
                for pat in args.patterns:
                    allow_patterns.append(f"{path.rstrip('/')}/{pat}")
        print(f"Downloading from {args.hf_repo}: {allow_patterns}")
        try:
            snapshot_download(
                repo_id=args.hf_repo,
                repo_type="dataset",
                allow_patterns=allow_patterns,
                local_dir=".",
            )
        except HfHubHTTPError as e:
            sys.exit(f"snapshot_download failed: {e}")
        # Report what landed
        for path in args.paths:
            p = Path(path)
            if p.is_dir():
                files = sorted(p.iterdir())
                print(f"  {p}: {len(files)} files")
            else:
                print(f"  {p}: {'present' if p.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
