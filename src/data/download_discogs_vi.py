"""Download Discogs-VI-YT metadata from Zenodo and emit a working subset CSV.

Source: https://zenodo.org/records/13983028

Outputs:
  data/raw/discogs_vi/Discogs-VI-YT-light-*.json   (raw metadata)
  data/splits/discogs_vi_subset.csv                 (cols: version_id, clique_id, youtube_id)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

ZENODO_RECORD_ID = 13983028
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def _fetch(url: str, dest: Path) -> None:
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
                print(f"\r  {read / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)", end="", file=sys.stderr)
        print(file=sys.stderr)
    tmp.rename(dest)


def _extract_light_json(zip_path: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        light = [
            n for n in names
            if "light" in n.lower() and n.lower().endswith(".json")
        ]
        if not light:
            light = [n for n in names if n.lower().endswith(".json") and "discogs-vi" in n.lower()]
        if not light:
            raise RuntimeError(
                f"No *light*.json or discogs-vi*.json inside {zip_path.name}. "
                f"Top-level entries: {names[:20]}"
            )
        member = light[0]
        print(f"Extracting {member}")
        zf.extract(member, path=out_dir)
        return out_dir / member


def fetch_metadata(out_dir: Path, keep_zip: bool) -> Path:
    existing = sorted(out_dir.rglob("*light*.json"))
    if existing:
        print(f"Light JSON already on disk: {existing[0]} — skipping fetch")
        return existing[0]

    print(f"Fetching Zenodo record {ZENODO_RECORD_ID} manifest")
    with urllib.request.urlopen(ZENODO_API) as r:
        record = json.load(r)

    files = record.get("files", [])
    # Prefer a direct light JSON if it ever appears at top level.
    direct = [f for f in files if "light" in f["key"].lower() and f["key"].lower().endswith(".json")]
    if direct:
        light = direct[0]
        dest = out_dir / light["key"]
        print(f"Downloading {light['key']} ({light['size'] / 1e6:.1f} MB)")
        _fetch(light["links"]["self"], dest)
        return dest

    # Otherwise the metadata lives inside main.zip.
    main_zip = next((f for f in files if f["key"].lower() == "main.zip"), None)
    if main_zip is None:
        raise RuntimeError(
            f"Could not find light JSON or main.zip in record {ZENODO_RECORD_ID}. "
            f"Files: {[f['key'] for f in files]}"
        )
    zip_dest = out_dir / "main.zip"
    print(f"Downloading main.zip ({main_zip['size'] / 1e6:.1f} MB)")
    _fetch(main_zip["links"]["self"], zip_dest)
    json_path = _extract_light_json(zip_dest, out_dir)
    if not keep_zip:
        try:
            zip_dest.unlink()
            print(f"  removed {zip_dest.name} after extraction")
        except OSError:
            pass
    return json_path


def parse_cliques(json_path: Path) -> list[dict]:
    """Return [{clique_id, version_id, youtube_id}, ...] from the light JSON.

    Schema observed in Discogs-VI-YT-20240701-light.json:
      { "<clique_id>": [ {"version_id", "track_title", "youtube_id"}, ... ], ... }
    """
    with json_path.open() as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Expected dict root in {json_path}, got {type(raw).__name__}")

    rows: list[dict] = []
    for clique_id, versions in raw.items():
        if not isinstance(versions, list):
            continue
        for v in versions:
            if not isinstance(v, dict):
                continue
            yid = v.get("youtube_id")
            vid = v.get("version_id")
            if not yid or not vid:
                continue
            if isinstance(yid, list):
                yid = yid[0] if yid else None
                if not yid:
                    continue
            rows.append({
                "clique_id": str(clique_id),
                "version_id": str(vid),
                "youtube_id": str(yid),
            })
    return rows


def sample_subset(
    rows: list[dict],
    n_cliques: int,
    min_versions: int,
    seed: int,
) -> list[dict]:
    by_clique: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_clique[r["clique_id"]].append(r)

    eligible = [(cid, vs) for cid, vs in by_clique.items() if len(vs) >= min_versions]
    print(f"  {len(by_clique)} cliques total, {len(eligible)} with >= {min_versions} versions")

    rng = random.Random(seed)
    rng.shuffle(eligible)
    picked = eligible[:n_cliques]

    out: list[dict] = []
    for _, vs in picked:
        # dedupe by youtube_id within a clique (some versions repeat IDs)
        seen = set()
        for v in vs:
            if v["youtube_id"] in seen:
                continue
            seen.add(v["youtube_id"])
            out.append(v)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["version_id", "clique_id", "youtube_id"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw/discogs_vi")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--n-cliques", type=int, default=2500)
    parser.add_argument("--min-versions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep main.zip after extracting the light JSON.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    splits_dir = Path(args.splits_dir)

    json_path = fetch_metadata(out_dir, keep_zip=args.keep_zip)

    print(f"Parsing {json_path.name}")
    rows = parse_cliques(json_path)
    print(f"  {len(rows)} (clique, version, youtube) rows parsed")
    if not rows:
        raise RuntimeError("Parser produced no rows — schema mismatch. Inspect the JSON manually.")

    subset = sample_subset(rows, args.n_cliques, args.min_versions, args.seed)
    out_csv = splits_dir / "discogs_vi_subset.csv"
    write_csv(subset, out_csv)
    n_cliques = len({r["clique_id"] for r in subset})
    print(f"Wrote {out_csv}: {len(subset)} tracks across {n_cliques} cliques")


if __name__ == "__main__":
    main()
