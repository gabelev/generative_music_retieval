"""yt-dlp crawler for the Discogs-VI-YT audio subset, streaming results to a
private Hugging Face dataset repo so local disk stays empty.

For each (version_id, youtube_id) in the subset CSV:
  1. yt-dlp downloads the audio (bestaudio, m4a/opus) into a tmp dir.
  2. ffmpeg transcodes to 24kHz mono WAV, center-clipped to 30 sec
     (MERT requirement).
  3. Upload the WAV to a private HF dataset repo at audio/<youtube_id>.wav.
  4. Delete BOTH the raw and the transcoded WAV from local disk.
  5. Append a row to the download log CSV (status + remote path).

Resume-safe: at startup, list remote audio/ files and skip those already there.
Run inside tmux/screen.

Env:
  HF_TOKEN must be set (or `huggingface-cli login` must have been run).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

CLIP_SECONDS = 30
TARGET_SR = 24000
HF_AUDIO_PREFIX = "audio"


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def ffprobe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, check=True, text=True,
        ).stdout
        return float(json.loads(out)["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, ValueError):
        return None


def yt_dlp_download(
    youtube_id: str,
    tmp_dir: Path,
    cookies_from_browser: str | None,
    js_runtime: str | None,
) -> tuple[Path | None, str]:
    """Returns (downloaded_path_or_None, error_message_or_empty)."""
    out_tmpl = str(tmp_dir / f"{youtube_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "-f", "bestaudio/best",
        "--retries", "3",
        "--fragment-retries", "3",
        "--no-progress",
        "--socket-timeout", "30",
        "-o", out_tmpl,
        "--print", "after_move:filepath",
        f"https://www.youtube.com/watch?v={youtube_id}",
    ]
    if js_runtime:
        cmd.extend(["--js-runtimes", js_runtime])
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        # keep error short for the log CSV
        err = err.splitlines()[-1][:200] if err else f"exit {proc.returncode}"
        return None, err
    path = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not path:
        matches = list(tmp_dir.glob(f"{youtube_id}.*"))
        if not matches:
            return None, "yt-dlp produced no file"
        path = str(matches[0])
    p = Path(path)
    if not p.exists():
        return None, f"path missing after download: {p}"
    return p, ""


def transcode_center_clip(src: Path, dst: Path) -> bool:
    duration = ffprobe_duration(src)
    if duration is None:
        return False
    start = max(0.0, (duration - CLIP_SECONDS) / 2.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.2f}",
        "-i", str(src),
        "-t", str(CLIP_SECONDS),
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def hf_upload(api: HfApi, repo_id: str, local: Path, path_in_repo: str) -> bool:
    for attempt in range(3):
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"crawl: {local.name}",
            )
            return True
        except HfHubHTTPError as e:
            if attempt == 2:
                print(f"  HF upload failed for {local.name}: {e}", file=sys.stderr)
                return False
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  HF upload exception for {local.name}: {e!r}", file=sys.stderr)
            return False
    return False


def list_existing(api: HfApi, repo_id: str) -> set[str]:
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except HfHubHTTPError as e:
        print(f"  warn: could not list repo {repo_id}: {e}", file=sys.stderr)
        return set()
    out: set[str] = set()
    prefix = f"{HF_AUDIO_PREFIX}/"
    for f in files:
        if f.startswith(prefix) and f.endswith(".wav"):
            yid = f[len(prefix):-len(".wav")]
            out.add(yid)
    return out


def process_one(
    row: dict,
    tmp_dir: Path,
    api: HfApi,
    repo_id: str,
    cookies_from_browser: str | None,
    js_runtime: str | None,
    sleep_min: float,
    sleep_max: float,
    keep_local: bool,
    audio_dir: Path,
) -> dict:
    yid = row["youtube_id"]
    path_in_repo = f"{HF_AUDIO_PREFIX}/{yid}.wav"
    work_wav = (audio_dir if keep_local else tmp_dir) / f"{yid}.wav"

    raw = None
    try:
        time.sleep(random.uniform(sleep_min, sleep_max))
        raw, err = yt_dlp_download(yid, tmp_dir, cookies_from_browser, js_runtime)
        if raw is None:
            status = "video_unavailable" if "unavailable" in err.lower() else "yt_dlp_failed"
            return {**row, "status": status, "hf_path": "", "error": err}
        if not transcode_center_clip(raw, work_wav):
            return {**row, "status": "ffmpeg_failed", "hf_path": "", "error": "ffmpeg transcode failed"}
        if not hf_upload(api, repo_id, work_wav, path_in_repo):
            return {**row, "status": "hf_upload_failed", "hf_path": "", "error": "HF upload failed"}
        return {**row, "status": "ok", "hf_path": path_in_repo, "error": ""}
    except Exception as e:
        return {**row, "status": "exception", "hf_path": "", "error": repr(e)}
    finally:
        try:
            if raw is not None and raw.exists():
                raw.unlink()
        except OSError:
            pass
        if not keep_local:
            try:
                if work_wav.exists():
                    work_wav.unlink()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-csv", default="data/splits/discogs_vi_subset.csv")
    parser.add_argument("--hf-repo", required=True,
                        help="Target HF dataset repo, e.g. <user>/discogs-vi-csi-subset")
    parser.add_argument("--log-csv", default="data/splits/discogs_vi_download_log.csv")
    parser.add_argument("--tmp-dir", default="data/raw/discogs_vi/_tmp",
                        help="Local scratch dir for in-flight downloads (cleaned per-track).")
    parser.add_argument("--keep-local-audio-dir", default=None,
                        help="If set, also keep transcoded WAVs locally under this dir.")
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Concurrent downloads. Keep low (2-4) to avoid bans.")
    parser.add_argument("--cookies-from-browser", default=None,
                        help="Pass to yt-dlp (e.g. 'chrome', 'safari') if hitting bot checks. "
                             "Note: Chrome's cookie DB can lock under concurrent access — if you "
                             "see widespread failures, omit this flag.")
    parser.add_argument("--js-runtime", default="node",
                        help="JS runtime for yt-dlp signature decryption (node|deno). "
                             "Recent yt-dlp + YouTube require this for most formats to resolve.")
    parser.add_argument("--sleep-min", type=float, default=0.5)
    parser.add_argument("--sleep-max", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, only crawl the first N rows (for smoke-testing).")
    parser.add_argument("--no-create-repo", action="store_true",
                        help="Skip the create_repo(exist_ok=True) call at startup.")
    args = parser.parse_args()

    for tool in ("yt-dlp", "ffmpeg", "ffprobe"):
        if not have(tool):
            sys.exit(f"Required tool not found on PATH: {tool}")
    if args.js_runtime and not have(args.js_runtime):
        print(f"warn: --js-runtime {args.js_runtime} not on PATH; yt-dlp may fail to resolve formats.",
              file=sys.stderr)
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if not token_file.exists():
            sys.exit("HF auth missing: set HF_TOKEN env var or run `huggingface-cli login`.")

    subset_csv = Path(args.subset_csv)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    keep_local = args.keep_local_audio_dir is not None
    audio_dir = Path(args.keep_local_audio_dir) if keep_local else tmp_dir
    if keep_local:
        audio_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    if not args.no_create_repo:
        try:
            api.create_repo(args.hf_repo, repo_type="dataset", private=True, exist_ok=True)
            print(f"HF repo ready: {args.hf_repo} (private dataset)")
        except HfHubHTTPError as e:
            sys.exit(f"create_repo failed for {args.hf_repo}: {e}")

    with subset_csv.open() as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    print(f"Loaded {len(rows)} rows from {subset_csv}")

    # Resume: union of local log success + remote repo files
    done_ids: set[str] = set()
    if log_path.exists():
        with log_path.open() as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    done_ids.add(r["youtube_id"])
    print(f"  {len(done_ids)} marked ok in local log")
    remote_ids = list_existing(api, args.hf_repo)
    print(f"  {len(remote_ids)} already in HF repo audio/")
    done_ids |= remote_ids

    todo = [r for r in rows if r["youtube_id"] not in done_ids]
    print(f"To download: {len(todo)}")

    log_exists = log_path.exists() and log_path.stat().st_size > 0
    log_f = log_path.open("a", newline="")
    fieldnames = ["version_id", "clique_id", "youtube_id", "status", "hf_path", "error"]
    writer = csv.DictWriter(log_f, fieldnames=fieldnames)
    if not log_exists:
        writer.writeheader()

    counts: dict[str, int] = {}
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [
                ex.submit(
                    process_one, r, tmp_dir, api, args.hf_repo,
                    args.cookies_from_browser, args.js_runtime,
                    args.sleep_min, args.sleep_max,
                    keep_local, audio_dir,
                )
                for r in todo
            ]
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                writer.writerow(rec)
                log_f.flush()
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
                if i % 25 == 0 or i == len(futures):
                    rate = i / max(1, time.time() - t0)
                    print(f"[{i}/{len(futures)}] {counts} ({rate:.2f}/s)")
    finally:
        log_f.close()
        try:
            for leftover in tmp_dir.iterdir():
                try:
                    leftover.unlink()
                except OSError:
                    pass
            tmp_dir.rmdir()
        except (OSError, FileNotFoundError):
            pass

    print(f"Done. Counts: {counts}")
    print(f"Log: {log_path}")
    print(f"Audio uploaded to: https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()
