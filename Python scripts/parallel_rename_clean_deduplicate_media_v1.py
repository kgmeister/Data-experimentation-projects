#!/usr/bin/env python3
# rename_clean_deduplicate_media_v29_parallel.py
# ============================================================
# Parallel media cleaner with safe parallel RENAMING.
# Per-folder invariant order:
#   1) Rename (parallel within folder; two-phase to avoid races)
#   2) Strip EXIF (parallel, images)
#   3) Compute hashes (parallel) → Compare (sequential)
#   4) Re-rename (sequential tidy)
# ============================================================

# --- Stdlib ---
import os, re, sys, shutil, argparse, tempfile, subprocess, time, uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util

# --- Auto-install third-party deps (Pillow, ImageHash, NumPy) ---
def _ensure_packages():
    need = {"PIL": "Pillow", "imagehash": "ImageHash", "numpy": "numpy"}
    missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"[setup] Installing: {', '.join(missing)}", file=sys.stderr)
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", *missing], check=True)
        except subprocess.CalledProcessError:
            subprocess.run([sys.executable, "-m", "pip", "install", "--user", "--upgrade", *missing], check=True)
_ensure_packages()

# --- Third-party (now safe to import) ---
from PIL import Image, ImageOps
import imagehash
import numpy as np  # noqa: F401

# --- Config / Constants ---
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS

DEFAULT_PHASH_THRESHOLD = 10
DEFAULT_DHASH_THRESHOLD_IMG = 2
DEFAULT_DHASH_THRESHOLD_VID = 7
DEFAULT_HASH_SIZE = 16
DEFAULT_NUM_FRAMES = 3

CPU = (os.cpu_count() or 4)
DEFAULT_WORKERS = max(4, CPU)
DEFAULT_RENAME_WORKERS = max(4, CPU)
DEFAULT_VIDEO_WORKERS = max(2, min(4, CPU // 2))

DUP_DIRNAME = "Duplicates"
CORRUPT_DIRNAME = "Corrupted"

RE_PATTERN_NUMBERED = re.compile(r"^(?P<base>.+?) \((?P<num>\d+)\)\.(?P<ext>[^.]+)$", re.I)

# --- Small FS helpers / logging ---
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def write_log(log_path: Path, lines: Iterable[str]) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        for ln in lines: f.write(ln.rstrip() + "\n")

def is_media(p: Path) -> bool: return p.is_file() and p.suffix.lower() in MEDIA_EXTS
def is_image(p: Path) -> bool: return p.suffix.lower() in IMAGE_EXTS
def is_video(p: Path) -> bool: return p.suffix.lower() in VIDEO_EXTS

def safe_move(src: Path, dst_dir: Path, new_name: Optional[str] = None) -> Path:
    ensure_dir(dst_dir)
    tgt = dst_dir / (new_name or src.name)
    if tgt.exists():
        stem, suf = tgt.stem, tgt.suffix
        i = 2
        while (cand := tgt.with_name(f"{stem}__{i}{suf}")).exists(): i += 1
        tgt = cand
    return src.rename(tgt)

# --- Data classes ---
@dataclass
class ImageHashes:
    phash: imagehash.ImageHash
    dhash: imagehash.ImageHash

@dataclass
class VideoHashes:
    frames: List[ImageHashes]

Hashes = Union[ImageHashes, VideoHashes]

@dataclass
class FileEntry:
    path: Path
    mtime: float
    is_image: bool
    is_video: bool

# ============================================================
# 1) INITIAL RENAME — PARALLEL (safe two-phase per folder)
# ============================================================
def _collect_folder_media(folder: Path) -> List[FileEntry]:
    out: List[FileEntry] = []
    files = [p for p in folder.iterdir() if is_media(p)]
    files.sort(key=lambda p: p.stat().st_mtime)
    for p in files:
        try:
            st = p.stat()
            out.append(FileEntry(path=p, mtime=st.st_mtime, is_image=is_image(p), is_video=is_video(p)))
        except Exception:
            continue
    return out

def _next_index_map(folder: Path, entries: List[FileEntry]) -> Dict[str, int]:
    base = folder.name
    highest: Dict[str, int] = {}
    for fe in entries:
        m = RE_PATTERN_NUMBERED.match(fe.path.name)
        if not m or m.group("base") != base: continue
        ext = "." + m.group("ext").lower()
        try: num = int(m.group("num"))
        except Exception: continue
        highest[ext] = max(highest.get(ext, 0), num)
    # next available
    return {ext: n + 1 for ext, n in highest.items()}

def _plan_initial_renames(folder: Path, entries: List[FileEntry]) -> Dict[Path, Path]:
    """Deterministic mapping for NON-conforming files → final numbered names."""
    base = folder.name
    next_idx = _next_index_map(folder, entries)
    nonconf = [fe for fe in entries if not (m := RE_PATTERN_NUMBERED.match(fe.path.name)) or m.group("base") != base]
    nonconf.sort(key=lambda fe: fe.mtime)  # oldest → newest
    mapping: Dict[Path, Path] = {}
    for fe in nonconf:
        ext = fe.path.suffix.lower()
        idx = next_idx.get(ext, 1)
        dst = folder / f"{base} ({idx}){ext}"
        # advance index for this ext
        next_idx[ext] = idx + 1
        mapping[fe.path] = dst
    return mapping

def _rename_to_tmp(src: Path) -> Path:
    tmp = src.with_name(f"__tmp_init_{uuid.uuid4().hex}_{src.name}")
    i = 2
    while tmp.exists():
        tmp = src.with_name(f"__tmp_init_{uuid.uuid4().hex}_{i}_{src.name}"); i += 1
    return src.rename(tmp)

def step1_rename_parallel(folder: Path, corrupt_dir: Path, workers: int) -> List[FileEntry]:
    entries = _collect_folder_media(folder)
    if not entries: return []
    plan = _plan_initial_renames(folder, entries)
    if not plan:     # nothing to do
        return entries

    # Phase A: to temp (parallel)
    tmp_to_final: Dict[Path, Path] = {}
    errors: List[Tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_rename_to_tmp, src): (src, dst) for src, dst in plan.items()}
        for fut in as_completed(futs):
            src, dst = futs[fut]
            try:
                tmp = fut.result()
                tmp_to_final[tmp] = dst
            except Exception as e:
                errors.append((src, str(e)))

    # Move failed sources to Corrupted
    for src, msg in errors:
        try: safe_move(src, corrupt_dir)
        except Exception: pass

    # Phase B: temp → final (parallel)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(lambda t, d: t.rename(d), t, d): (t, d) for t, d in tmp_to_final.items()}
        for fut in as_completed(futs):
            t, d = futs[fut]
            try:
                fut.result()
            except Exception:
                # best-effort recovery: push temp to corrupted
                try: safe_move(t, corrupt_dir)
                except Exception: pass

    # Re-collect (names changed)
    return _collect_folder_media(folder)

# ============================================================
# 2) STRIP EXIF — PARALLEL (images)
# ============================================================
def _strip_exif_one(path: Path) -> Tuple[Path, Optional[str]]:
    if path.suffix.lower() == ".gif":  # keep animations intact
        return path, None
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            fmt = (im.format or path.suffix.replace(".", "")).upper()
            if fmt in ("JPG", "JPEG"):
                im.convert("RGB").save(path, format="JPEG", quality=95, optimize=True)
            elif fmt == "PNG":
                im.save(path, format="PNG", optimize=True)
            elif fmt == "WEBP":
                im.save(path, format="WEBP", quality=95, method=6)
            else:
                im.save(path)
        return path, None
    except Exception as e:
        return path, str(e)

def step2_strip_exif_parallel(entries: List[FileEntry], corrupt_dir: Path, workers: int) -> None:
    imgs = [fe.path for fe in entries if fe.is_image]
    if not imgs: return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_strip_exif_one, p) for p in imgs]
        for fut in as_completed(futs):
            p, err = fut.result()
            if err:
                try: safe_move(p, corrupt_dir)
                except Exception: pass

# ============================================================
# 3) HASH (PARALLEL) → COMPARE (SEQUENTIAL)
# ============================================================
def _compute_image_hashes(path: Path, hash_size: int) -> Optional[ImageHashes]:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return ImageHashes(
                phash=imagehash.phash(im, hash_size=hash_size),
                dhash=imagehash.dhash(im, hash_size=hash_size)
            )
    except Exception:
        return None

def _probe_duration(ffprobe: str, path: Path) -> Optional[float]:
    try:
        out = subprocess.check_output(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stderr=subprocess.STDOUT
        )
        return float(out.strip())
    except Exception:
        return None

def _extract_frame_at(ffmpeg: str, src: Path, t: float, out_dir: Path, idx: int) -> Optional[Path]:
    out_path = out_dir / f"frame_{idx:02d}.png"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(src),
             "-frames:v", "1", "-q:v", "2", str(out_path)],
            check=True
        )
        return out_path if out_path.exists() else None
    except Exception:
        return None

def _compute_video_hashes(path: Path, hash_size: int, num_frames: int, ffmpeg: str, ffprobe: str) -> Optional[VideoHashes]:
    dur = _probe_duration(ffprobe, path)
    if not dur or dur <= 0: return None
    times = [(dur * (i + 1) / (num_frames + 1)) for i in range(num_frames)]
    tmp = Path(tempfile.mkdtemp(prefix="vidframes_"))
    frames: List[ImageHashes] = []
    try:
        for i, t in enumerate(times):
            fp = _extract_frame_at(ffmpeg, path, t, tmp, i)
            if not fp: return None
            ih = _compute_image_hashes(fp, hash_size)
            if not ih: return None
            frames.append(ih)
        return VideoHashes(frames=frames)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def _compute_hashes_worker(fe: FileEntry, hash_size: int, num_frames: int, ffmpeg: Optional[str], ffprobe: Optional[str]) -> Tuple[Path, Optional[Hashes]]:
    if fe.is_image: return fe.path, _compute_image_hashes(fe.path, hash_size)
    if fe.is_video and ffmpeg and ffprobe:
        return fe.path, _compute_video_hashes(fe.path, hash_size, num_frames, ffmpeg, ffprobe)
    return fe.path, None

def step3_compute_hashes_parallel(entries: List[FileEntry], hash_size: int, num_frames: int, workers: int, video_workers: int) -> Dict[Path, Optional[Hashes]]:
    ffmpeg, ffprobe = which("ffmpeg"), which("ffprobe")
    if not ffmpeg or not ffprobe:
        print("[warn] ffmpeg/ffprobe not found → video hashing skipped.", file=sys.stderr)
    out: Dict[Path, Optional[Hashes]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_compute_hashes_worker, fe, hash_size, num_frames, ffmpeg, ffprobe) for fe in entries]
        for fut in as_completed(futs):
            p, h = fut.result()
            out[p] = h
    return out

def _images_similar(a: ImageHashes, b: ImageHashes, pth: int, dth: int) -> Tuple[bool, int, int]:
    pd, dd = int(a.phash - b.phash), int(a.dhash - b.dhash)
    return (pd <= pth and dd <= dth), pd, dd

def _videos_similar(a: VideoHashes, b: VideoHashes, pth: int, dth: int) -> Tuple[bool, List[Tuple[int, int]]]:
    dists: List[Tuple[int, int]] = []
    for fa, fb in zip(a.frames, b.frames):
        pd, dd = int(fa.phash - fb.phash), int(fa.dhash - fb.dhash)
        dists.append((pd, dd))
        if not (pd <= pth and dd <= dth): return False, dists
    return True, dists

def step3_compare_and_dedupe_sequential(folder: Path, entries: List[FileEntry], hashes: Dict[Path, Optional[Hashes]],
                                        dup_dir: Path, log_path: Path, pth: int, dth_img: int, dth_vid: int) -> None:
    by_ext: Dict[str, List[FileEntry]] = {}
    for fe in entries: by_ext.setdefault(fe.path.suffix.lower(), []).append(fe)

    for ext, group in by_ext.items():
        group.sort(key=lambda fe: fe.mtime)
        kept: List[FileEntry] = []
        for fe in group:
            h = hashes.get(fe.path)
            if h is None: kept.append(fe); continue
            is_dup = False; lines: List[str] = []
            for base in kept:
                hb = hashes.get(base.path)
                if hb is None: continue
                if fe.is_image and isinstance(h, ImageHashes) and isinstance(hb, ImageHashes):
                    ok, pd, dd = _images_similar(h, hb, pth, dth_img)
                    if ok:
                        is_dup = True
                        lines.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | IMG DUP | base={base.path} | dup={fe.path} | phash={pd} dhash={dd}")
                        break
                elif fe.is_video and isinstance(h, VideoHashes) and isinstance(hb, VideoHashes):
                    ok, dists = _videos_similar(h, hb, pth, dth_vid)
                    if ok:
                        is_dup = True
                        ds = " ".join([f"[p{p}/d{d}]" for p, d in dists])
                        lines.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | VID DUP | base={base.path} | dup={fe.path} | {ds}")
                        break
            if is_dup:
                try:
                    safe_move(fe.path, dup_dir, new_name=f"{folder.name}_duplicate_{fe.path.name}")
                    if lines: write_log(log_path, lines)
                except Exception as e:
                    write_log(log_path, [f"{time.strftime('%Y-%m-%d %H:%M:%S')} | ERROR moving dup | {fe.path} | {e}"])
            else:
                kept.append(fe)

# ============================================================
# 4) RE-RENAME — SEQUENTIAL (tidy numbering per ext)
# ============================================================
def _two_phase_rename(mapping: Dict[Path, Path]) -> None:
    tmp_map: Dict[Path, Path] = {}
    for src, dst in mapping.items():
        if src == dst: continue
        tmp = src.with_name(f"__tmp__{src.name}"); i = 2
        while tmp.exists(): tmp = src.with_name(f"__tmp__{i}__{src.name}"); i += 1
        src.rename(tmp); tmp_map[tmp] = dst
    for tmp, dst in tmp_map.items():
        if dst.exists(): dst.unlink()
        tmp.rename(dst)

def step4_rerename_tidy(folder: Path) -> None:
    entries = _collect_folder_media(folder)
    if not entries: return
    by_ext: Dict[str, List[FileEntry]] = {}
    for fe in entries: by_ext.setdefault(fe.path.suffix.lower(), []).append(fe)
    base = folder.name
    mapping: Dict[Path, Path] = {}
    for ext, group in by_ext.items():
        group.sort(key=lambda fe: fe.mtime)
        for i, fe in enumerate(group, 1):
            final = folder / f"{base} ({i}){ext}"
            if fe.path != final: mapping[fe.path] = final
    if mapping: _two_phase_rename(mapping)

# ============================================================
# Folder discovery / Orchestration
# ============================================================
def iter_media_folders(root: Path) -> List[Path]:
    folders: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_dir(): continue
        if p.name in (DUP_DIRNAME, CORRUPT_DIRNAME): continue
        try:
            has_media = any(is_media(c) for c in p.iterdir() if c.is_file())
        except Exception:
            has_media = False
        if has_media: folders.append(p)
    folders.sort(key=lambda x: str(x).lower())
    return folders

def main():
    ap = argparse.ArgumentParser(description="Parallel media cleaner & deduplicator (safe parallel rename).")
    ap.add_argument("--root-dir", type=str, default="Content")
    ap.add_argument("--phash-threshold", type=int, default=DEFAULT_PHASH_THRESHOLD)
    ap.add_argument("--dhash-img-threshold", type=int, default=DEFAULT_DHASH_THRESHOLD_IMG)
    ap.add_argument("--dhash-vid-threshold", type=int, default=DEFAULT_DHASH_THRESHOLD_VID)
    ap.add_argument("--hash-size", type=int, default=DEFAULT_HASH_SIZE)
    ap.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers for EXIF/hash.")
    ap.add_argument("--rename-workers", type=int, default=DEFAULT_RENAME_WORKERS, help="Parallel workers for initial rename.")
    ap.add_argument("--video-workers", type=int, default=DEFAULT_VIDEO_WORKERS)
    args = ap.parse_args()

    root = Path(args.root_dir).resolve()
    if not root.exists():
        print(f"[error] Root not found: {root}", file=sys.stderr); sys.exit(1)

    dup_dir = ensure_dir(root / DUP_DIRNAME)
    corrupt_dir = ensure_dir(root / CORRUPT_DIRNAME)
    log_path = root / "deduplication_log.txt"

    # Include root itself if it contains media
    folders: List[Path] = []
    try:
        if any(is_media(p) for p in root.iterdir() if p.is_file()):
            folders.append(root)
    except Exception:
        pass
    folders.extend(iter_media_folders(root))
    if not folders:
        print("[info] No media folders found."); return

    print(f"[info] Processing {len(folders)} folder(s) under: {root}")
    for folder in folders:
        if folder.name in (DUP_DIRNAME, CORRUPT_DIRNAME): continue
        print(f"\n[folder] {folder}")

        # 1) Initial rename (parallel within folder; two-phase)
        entries = step1_rename_parallel(folder, corrupt_dir, workers=args.rename_workers)
        if not entries: print("  - No media after rename."); continue

        # 2) Strip EXIF (parallel)
        step2_strip_exif_parallel(entries, corrupt_dir, workers=args.workers)

        # Re-collect (some may have moved to Corrupted)
        entries = _collect_folder_media(folder)
        if not entries: print("  - No media after EXIF step."); continue

        # 3a) Hash (parallel)
        hashes = step3_compute_hashes_parallel(entries, args.hash_size, args.num_frames, args.workers, args.video_workers)

        # 3b) Compare (sequential) + move dupes
        step3_compare_and_dedupe_sequential(
            folder, entries, hashes, dup_dir, log_path,
            pth=args.phash_threshold, dth_img=args.dhash_img_threshold, dth_vid=args.dhash_vid_threshold
        )

        # 4) Re-rename tidy (sequential)
        step4_rerename_tidy(folder)

    print("\n[done] Processing complete.")

if __name__ == "__main__":
    main()
