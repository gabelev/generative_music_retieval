"""yt-dlp crawler for the Discogs-VI-YT audio subset.

Pipeline per row:
  1. yt-dlp downloads the audio (bestaudio, m4a/opus) into a tmp dir.
  2. ffmpeg transcodes to 24kHz mono WAV, center-clipped to 30 sec.
  3. WAV is staged into a local "to-upload" dir.

Then, in batches of --upload-batch-size, the staged folder is uploaded to a
private HF dataset repo via upload_folder() — ONE COMMIT per batch — and the
local staged files are deleted. Free-tier HF dataset repos are limited to
128 commits/hour, so per-file uploads blow the rate limit at ~128 tracks.
Batching keeps total commits ~120 for a 12K-track crawl.

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


def download_one(
    row: dict,
    tmp_dir: Path,
    stage_dir: Path,
    cookies_from_browser: str | None,
    js_runtime: str | None,
    sleep_min: float,
    sleep_max: float,
) -> dict:
    """Download + transcode one row. Stages the WAV at stage_dir/<yid>.wav.

    Returns a dict carrying the row fields plus status, staged_path, error.
    Does NOT upload to HF — batched upload happens in main().
    """
    yid = row["youtube_id"]
    staged = stage_dir / f"{yid}.wav"

    if staged.exists() and staged.stat().st_size > 0:
        return {**row, "status": "staged", "staged_path": str(staged), "error": ""}

    raw = None
    try:
        time.sleep(random.uniform(sleep_min, sleep_max))
        raw, err = yt_dlp_download(yid, tmp_dir, cookies_from_browser, js_runtime)
        if raw is None:
            status = "video_unavailable" if "unavailable" in err.lower() else "yt_dlp_failed"
            return {**row, "status": status, "staged_path": "", "error": err}
        if not transcode_center_clip(raw, staged):
            return {**row, "status": "ffmpeg_failed", "staged_path": "", "error": "ffmpeg transcode failed"}
        return {**row, "status": "staged", "staged_path": str(staged), "error": ""}
    except Exception as e:
        return {**row, "status": "exception", "staged_path": "", "error": repr(e)}
    finally:
        try:
            if raw is not None and raw.exists():
                raw.unlink()
        except OSError:
            pass


def upload_batch(
    api: HfApi,
    repo_id: str,
    stage_dir: Path,
    staged_yids: list[str],
    chunk_idx: int,
) -> tuple[bool, str]:
    """One commit: push stage_dir/*.wav to <repo>/audio/. Returns (ok, error)."""
    if not staged_yids:
        return True, ""
    for attempt in range(3):
        try:
            api.upload_folder(
                folder_path=str(stage_dir),
                path_in_repo=HF_AUDIO_PREFIX,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"crawl batch {chunk_idx}: {len(staged_yids)} wavs",
                allow_patterns=["*.wav"],
            )
            return True, ""
        except HfHubHTTPError as e:
            msg = str(e)
            # Detect rate limit and back off for the suggested duration.
            wait_s = 30 * (attempt + 1)
            if "Too Many Requests" in msg or "429" in msg:
                # Parse "Retry after N seconds" or "in about 1 hour" if present.
                wait_s = 60 if "in about 1 hour" not in msg else 3600
                print(f"  HF rate-limited, sleeping {wait_s}s before retry {attempt + 1}/3",
                      file=sys.stderr)
            else:
                print(f"  HF upload_folder error (attempt {attempt + 1}/3): {msg[:200]}",
                      file=sys.stderr)
            if attempt == 2:
                return False, msg[:300]
            time.sleep(wait_s)
        except Exception as e:
            print(f"  HF upload_folder exception: {e!r}", file=sys.stderr)
            return False, repr(e)[:300]
    return False, "exhausted retries"


def cleanup_staged(stage_dir: Path, yids: list[str]) -> None:
    for yid in yids:
        p = stage_dir / f"{yid}.wav"
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-csv", default="data/splits/discogs_vi_subset.csv")
    parser.add_argument("--hf-repo", required=True,
                        help="Target HF dataset repo, e.g. <user>/discogs-vi-csi-subset")
    parser.add_argument("--log-csv", default="data/splits/discogs_vi_download_log.csv")
    parser.add_argument("--tmp-dir", default="data/raw/discogs_vi/_tmp")
    parser.add_argument("--stage-dir", default="data/raw/discogs_vi/_stage",
                        help="Local staging dir; cleared after each batch upload.")
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Concurrent downloads. Keep low (2-4) to avoid bans.")
    parser.add_argument("--upload-batch-size", type=int, default=100,
                        help="WAVs per HF commit. HF free tier = 128 commits/hour on "
                             "dataset repos; 100 keeps us safely under that for a 12K crawl.")
    parser.add_argument("--cookies-from-browser", default=None,
                        help="Pass to yt-dlp (e.g. 'chrome', 'safari') if hitting bot checks. "
                             "Chrome's cookie DB can lock under concurrent access — omit if "
                             "you see widespread failures.")
    parser.add_argument("--js-runtime", default="node",
                        help="JS runtime for yt-dlp signature decryption (node|deno).")
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
    stage_dir = Path(args.stage_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
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
    if not todo:
        print("Nothing to do.")
        return

    log_exists = log_path.exists() and log_path.stat().st_size > 0
    log_f = log_path.open("a", newline="")
    fieldnames = ["version_id", "clique_id", "youtube_id", "status", "hf_path", "error"]
    writer = csv.DictWriter(log_f, fieldnames=fieldnames)
    if not log_exists:
        writer.writeheader()

    counts: dict[str, int] = {}
    t0 = time.time()
    batch_size = max(1, args.upload_batch_size)
    n_batches = (len(todo) + batch_size - 1) // batch_size
    try:
        for chunk_idx in range(n_batches):
            batch = todo[chunk_idx * batch_size : (chunk_idx + 1) * batch_size]

            # 1. Parallel download + transcode (no uploads here)
            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                futures = [
                    ex.submit(
                        download_one, r, tmp_dir, stage_dir,
                        args.cookies_from_browser, args.js_runtime,
                        args.sleep_min, args.sleep_max,
                    )
                    for r in batch
                ]
                results = [f.result() for f in as_completed(futures)]

            staged = [r for r in results if r["status"] == "staged"]
            staged_yids = [r["youtube_id"] for r in staged]

            # 2. One HF commit for the whole batch
            upload_ok, upload_err = upload_batch(
                api, args.hf_repo, stage_dir, staged_yids, chunk_idx
            )

            # 3. Mark each row's final status and write log
            for r in results:
                if r["status"] == "staged":
                    if upload_ok:
                        r["status"] = "ok"
                        r["hf_path"] = f"{HF_AUDIO_PREFIX}/{r['youtube_id']}.wav"
                    else:
                        r["status"] = "hf_upload_failed"
                        r["hf_path"] = ""
                        r["error"] = upload_err
                else:
                    r.setdefault("hf_path", "")
                r.pop("staged_path", None)
                writer.writerow({k: r.get(k, "") for k in fieldnames})
                counts[r["status"]] = counts.get(r["status"], 0) + 1
            log_f.flush()

            # 4. Cleanup staged WAVs (uploaded successfully or not — failed
            # batches will be retried as fresh downloads on next run, which is
            # simpler than tracking partial-staged state across runs)
            cleanup_staged(stage_dir, staged_yids)

            done = (chunk_idx + 1) * batch_size
            done = min(done, len(todo))
            rate = done / max(1, time.time() - t0)
            print(f"[batch {chunk_idx + 1}/{n_batches} | "
                  f"{done}/{len(todo)}] {counts} ({rate:.2f}/s)")

    finally:
        log_f.close()
        for d in (tmp_dir, stage_dir):
            try:
                for leftover in d.iterdir():
                    try:
                        leftover.unlink()
                    except OSError:
                        pass
                d.rmdir()
            except (OSError, FileNotFoundError):
                pass

    print(f"Done. Counts: {counts}")
    print(f"Log: {log_path}")
    print(f"Audio uploaded to: https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()
