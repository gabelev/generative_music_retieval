"""Download Covers80 (Ellis 2007) and build the clique mapping CSV.

  - Tarball: http://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tgz
    (~156 MB; mtime 2007-08-08, static — no link rot.)

Outputs:
  data/raw/covers80/covers80.tgz                      (downloaded archive)
  data/raw/covers80/covers32k/<Artist+Song>/*.mp3     (extracted audio)
  data/raw/covers80/list1.txt, list2.txt              (paired clique lists)
  data/splits/covers80_cliques.csv                    (track_id, clique_id, filepath, version)

Clique semantics: list1.txt line i and list2.txt line i are two versions
of the same composition. We build clique_id = "C80-<i>" and assign
version "A" / "B" accordingly.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tgz"


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists: {dest} ({dest.stat().st_size / 1e6:.1f} MB) — skipping")
        return
    print(f"  GET {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        read = 0
        chunk = 1 << 20
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            read += len(buf)
            if total:
                pct = 100 * read / total
                print(f"\r  {read / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                      end="", file=sys.stderr)
        print(file=sys.stderr)
    tmp.rename(dest)


def extract(tar_path: Path, out_dir: Path) -> Path:
    """Extract the tarball and return the directory containing covers32k/ + lists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        members = tf.getmembers()
        # Detect top-level dir name (usually 'covers80/')
        top_levels = {m.name.split("/", 1)[0] for m in members if m.name}
        top = sorted(top_levels)[0] if top_levels else "covers80"
        already = out_dir / top
        if already.exists() and any(already.iterdir()):
            print(f"  already extracted at {already} — skipping")
            return already
        # Strip leading path components so we don't get path traversal issues.
        # Modern Python (3.12+) requires a filter for tar extraction.
        tf.extractall(out_dir, filter="data")
    return out_dir / top


def find_lists(root: Path) -> tuple[Path, Path]:
    # Original tarball uses .list; some mirrors / forks use .txt.
    list1 = next(iter(list(root.rglob("list1.list")) + list(root.rglob("list1.txt"))), None)
    list2 = next(iter(list(root.rglob("list2.list")) + list(root.rglob("list2.txt"))), None)
    if list1 is None or list2 is None:
        raise RuntimeError(f"Could not find list1.* / list2.* under {root}")
    return list1, list2


def find_audio_root(root: Path) -> Path:
    for cand in [root / "covers32k", root / "covers", root]:
        if cand.exists() and any(p.is_dir() for p in cand.iterdir()):
            return cand
    raise RuntimeError(f"No audio subdir (covers32k/...) found under {root}")


def resolve_track(audio_root: Path, list_entry: str) -> Path | None:
    """list_entry is something like 'The_Beatles+Come_Together/beatles' (no extension).
    Find the matching mp3 file under audio_root.
    """
    entry = list_entry.strip()
    if not entry:
        return None
    # Try direct: audio_root/<entry>.mp3
    for ext in (".mp3", ".MP3", ".wav", ".WAV"):
        p = audio_root / f"{entry}{ext}"
        if p.exists():
            return p
    # Try with normalized basename (some lists use slightly different separators)
    parts = entry.split("/")
    if len(parts) == 2:
        clique_dir, base = parts
        for d in audio_root.iterdir():
            if d.is_dir() and d.name == clique_dir:
                for f in d.iterdir():
                    if f.stem == base and f.suffix.lower() in (".mp3", ".wav"):
                        return f
    return None


def build_cliques_csv(
    list1: Path, list2: Path, audio_root: Path, out_csv: Path,
) -> tuple[int, int]:
    lines1 = [ln.rstrip("\n") for ln in list1.read_text().splitlines() if ln.strip()]
    lines2 = [ln.rstrip("\n") for ln in list2.read_text().splitlines() if ln.strip()]
    if len(lines1) != len(lines2):
        print(f"  warn: list1 has {len(lines1)} entries, list2 has {len(lines2)} — pairing up to min.",
              file=sys.stderr)
    n = min(len(lines1), len(lines2))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = 0
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "clique_id", "filepath", "version"])
        w.writeheader()
        for i in range(n):
            clique_id = f"C80-{i:03d}"
            for version, entry in (("A", lines1[i]), ("B", lines2[i])):
                p = resolve_track(audio_root, entry)
                if p is None:
                    missing += 1
                    continue
                track_id = f"{clique_id}-{version}"
                w.writerow({
                    "track_id": track_id,
                    "clique_id": clique_id,
                    "filepath": str(p.resolve()),
                    "version": version,
                })
                written += 1
    return written, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw/covers80")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--keep-tarball", action="store_true",
                        help="Keep covers80.tgz after extraction.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    splits_dir = Path(args.splits_dir)

    tarball = out_dir / "covers80.tgz"
    fetch(args.url, tarball)

    print(f"Extracting {tarball.name} into {out_dir}")
    extracted_root = extract(tarball, out_dir)
    print(f"  extracted root: {extracted_root}")

    list1, list2 = find_lists(extracted_root)
    audio_root = find_audio_root(extracted_root)
    print(f"  lists: {list1.name}, {list2.name}")
    print(f"  audio root: {audio_root}")

    out_csv = splits_dir / "covers80_cliques.csv"
    n_written, n_missing = build_cliques_csv(list1, list2, audio_root, out_csv)
    print(f"Wrote {out_csv}: {n_written} tracks, {n_missing} missing file references")

    if not args.keep_tarball:
        try:
            tarball.unlink()
            print(f"  removed {tarball.name} after extraction")
        except OSError:
            pass


if __name__ == "__main__":
    main()
