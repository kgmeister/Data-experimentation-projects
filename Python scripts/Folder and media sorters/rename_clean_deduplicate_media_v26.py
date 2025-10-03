import os
from PIL import Image
import imagehash
import mimetypes
import shutil
import subprocess
import tempfile
import re
import atexit
import signal
import sys
import argparse

# Global list to track temporary directories
temp_dirs = []

def cleanup_temp_dirs():
    """Clean up all temporary directories on script exit."""
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory on exit: {temp_dir}")
            except Exception as e:
                print(f"Error cleaning up temporary directory {temp_dir} on exit: {e}")

# Register cleanup function to run on script exit or interruption
atexit.register(cleanup_temp_dirs)
def signal_handler(sig, frame):
    cleanup_temp_dirs()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def move_to_corrupted(corrupted_dir, file_path, reason):
    """Move a file to the Corrupted folder with a reason."""
    try:
        os.makedirs(corrupted_dir, exist_ok=True)
        corrupted_path = os.path.join(corrupted_dir, os.path.basename(file_path))
        shutil.move(file_path, corrupted_path)
        print(f"Moved to Corrupted due to {reason}: {file_path} -> {corrupted_path}")
        return True
    except Exception as e:
        print(f"Error moving {file_path} to Corrupted: {e}")
        return False

def remove_exif_image(image_path, corrupted_dir):
    """Remove EXIF data from an image file."""
    try:
        image = Image.open(image_path)
        image.save(image_path, exif=b'')
        image.close()
        print(f"Removed EXIF data from: {image_path}")
    except Exception as e:
        print(f"Error removing EXIF from {image_path}: {e}")
        move_to_corrupted(corrupted_dir, image_path, "EXIF removal failure")

def get_image_hash(image_path, corrupted_dir):
    """Compute both perceptual hash (phash) and difference hash (dhash) for an image with 16x16 hash size."""
    try:
        with Image.open(image_path) as img:
            phash_value = imagehash.phash(img, hash_size=16)
            dhash_value = imagehash.dhash(img, hash_size=16)
            print(f"pHash (16x16) for {image_path}: {str(phash_value)}")
            print(f"dHash (16x16) for {image_path}: {str(dhash_value)}")
            return (phash_value, dhash_value)
    except Exception as e:
        print(f"Error hashing {image_path}: {e}")
        move_to_corrupted(corrupted_dir, image_path, "hashing failure")
        return None

def get_video_frame_hashes(video_path, corrupted_dir, num_frames=3):
    """Extract frames from a video and compute phash and dhash for each with 16x16 hash size."""
    temp_dir = None
    frame_paths = []
    try:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise FileNotFoundError("FFmpeg not found in PATH. Please install FFmpeg and add it to PATH.")

        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")
        temp_dirs.append(temp_dir)

        # Get video duration
        result = subprocess.run([ffmpeg_path, "-i", video_path], capture_output=True, text=True)
        duration = 10  # Fallback duration
        match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})", result.stderr)
        if match:
            h, m, s = map(float, match.groups())
            duration = h * 3600 + m * 60 + s
        else:
            print(f"Could not parse duration for {video_path}: {result.stderr}")

        # Extract frames
        interval = duration / (num_frames + 1)
        for i in range(1, num_frames + 1):
            output_path = os.path.join(temp_dir, f"frame_{i}.png")
            try:
                result = subprocess.run([
                    ffmpeg_path, "-i", video_path, "-ss", str(i * interval),
                    "-vframes", "1", "-q:v", "2", output_path
                ], check=True, capture_output=True, text=True)
                if os.path.exists(output_path):
                    frame_paths.append(output_path)
                    print(f"Created temporary frame: {output_path}")
                else:
                    print(f"Failed to create temporary frame: {output_path}")
                    print(f"FFmpeg stderr: {result.stderr}")
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg error extracting frame {i} from {video_path}: {e}")
                print(f"FFmpeg stderr: {e.stderr}")

        # Compute phash and dhash for frames
        frame_hashes = []
        for frame_path in frame_paths:
            try:
                with Image.open(frame_path) as img:
                    phash_value = imagehash.phash(img, hash_size=16)
                    dhash_value = imagehash.dhash(img, hash_size=16)
                    frame_hashes.append((phash_value, dhash_value))
                    print(f"pHash (16x16) for {frame_path}: {str(phash_value)}")
                    print(f"dHash (16x16) for {frame_path}: {str(dhash_value)}")
            except Exception as e:
                print(f"Error hashing frame {frame_path}: {e}")
        if not frame_hashes:
            move_to_corrupted(corrupted_dir, video_path, "no valid frame hashes")
            return None
        return frame_hashes
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        move_to_corrupted(corrupted_dir, video_path, "video processing failure")
        return None
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                for frame_path in frame_paths:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                        print(f"Deleted temporary frame: {frame_path}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"Deleted temporary directory: {temp_dir}")
                if temp_dir in temp_dirs:
                    temp_dirs.remove(temp_dir)
            except Exception as e:
                print(f"Error deleting temporary directory {temp_dir}: {e}")

def get_highest_existing_number(folder_path, folder, extension):
    """Find the highest number among correctly named files in the folder."""
    max_number = 0
    for filename in os.listdir(folder_path):
        match = re.match(rf"{re.escape(folder)} \((\d+)\){re.escape(extension)}$", filename)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)
    return max_number

def get_next_available_number(folder_path, folder, extension, start_number):
    """Find the next available number starting from start_number."""
    i = start_number + 1
    while True:
        new_filename = f"{folder} ({i}){extension}"
        new_path = os.path.join(folder_path, new_filename)
        if not os.path.exists(new_path):
            return i
        i += 1

def process_folder(root_dir, phash_threshold=10, dhash_threshold=2):
    """Process all media files in subfolders recursively."""
    print(f"Using thresholds: phash_threshold={phash_threshold} (images), {phash_threshold} (videos); "
          f"dhash_threshold={dhash_threshold} (images), 7 (videos)")
    media_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.mp4', '.mov', '.avi'}
    duplicates_dir = os.path.join(root_dir, "Duplicates")
    corrupted_dir = os.path.join(root_dir, "Corrupted")
    os.makedirs(duplicates_dir, exist_ok=True)
    os.makedirs(corrupted_dir, exist_ok=True)

    for folder_path, _, files in os.walk(root_dir):
        if os.path.basename(folder_path) in ("Duplicates", "Corrupted"):
            continue
        folder = os.path.basename(folder_path)
        print(f"\nProcessing folder: {folder_path}")
        media_files = [f for f in files if os.path.splitext(f)[1].lower() in media_extensions]
        media_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

        # Step 1: Rename all files
        renamed_files = []
        for filename in media_files:
            old_path = os.path.join(folder_path, filename)
            extension = os.path.splitext(filename)[1].lower()
            match = re.match(rf"{re.escape(folder)} \((\d+)\){re.escape(extension)}$", filename)
            if match:
                renamed_files.append(filename)
                print(f"Skipped renaming (already correct): {old_path}")
                continue
            # Start numbering from the highest existing number for this extension
            highest_number = get_highest_existing_number(folder_path, folder, extension)
            file_number = get_next_available_number(folder_path, folder, extension, highest_number)
            new_filename = f"{folder} ({file_number}){extension}"
            new_path = os.path.join(folder_path, new_filename)
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
                renamed_files.append(new_filename)
            except Exception as e:
                print(f"Error renaming {old_path}: {e}")
                move_to_corrupted(corrupted_dir, old_path, "renaming failure")
                continue

        # Step 2: Remove EXIF data from images
        for filename in renamed_files:
            file_path = os.path.join(folder_path, filename)
            if not os.path.exists(file_path):
                continue
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith('image'):
                remove_exif_image(file_path, corrupted_dir)

        # Step 3: Deduplicate files using Hamming distance
        hash_dict = {}
        for filename in renamed_files:
            file_path = os.path.join(folder_path, filename)
            if not os.path.exists(file_path):
                continue
            extension = os.path.splitext(filename)[1].lower()
            mime_type, _ = mimetypes.guess_type(file_path)

            # Compute hash based on file type
            file_hashes = None
            if mime_type and mime_type.startswith('image'):
                hash_value = get_image_hash(file_path, corrupted_dir)
                file_hashes = [hash_value] if hash_value else None
            elif mime_type and mime_type.startswith('video'):
                file_hashes = get_video_frame_hashes(file_path, corrupted_dir)
            if not file_hashes:
                continue

            # Check for duplicates
            is_duplicate = False
            for existing_hashes in hash_dict:
                if len(file_hashes) != len(existing_hashes):
                    continue
                all_similar = True
                # Set thresholds based on file type
                current_phash_threshold = phash_threshold  # 10 for both images and videos
                current_dhash_threshold = 7 if mime_type and mime_type.startswith('video') else dhash_threshold
                for new_hash, existing_hash in zip(file_hashes, existing_hashes):
                    new_phash, new_dhash = new_hash
                    existing_phash, existing_dhash = existing_hash
                    phash_distance = new_phash - existing_phash
                    dhash_distance = new_dhash - existing_dhash
                    if phash_distance > current_phash_threshold or dhash_distance > current_dhash_threshold:
                        all_similar = False
                        break
                    with open(os.path.join(root_dir, "deduplication_log.txt"), "a") as log_file:
                        log_file.write(f"pHash (16x16) for {file_path}: {str(new_phash)}\n")
                        log_file.write(f"dHash (16x16) for {file_path}: {str(new_dhash)}\n")
                        log_file.write(f"pHash (16x16) for {hash_dict[existing_hashes]}: {str(existing_phash)}\n")
                        log_file.write(f"dHash (16x16) for {hash_dict[existing_hashes]}: {str(existing_dhash)}\n")
                        log_file.write(f"{file_path} vs {hash_dict[existing_hashes]} calculated pHash distance: {phash_distance}\n")
                        log_file.write(f"{file_path} vs {hash_dict[existing_hashes]} calculated dHash distance: {dhash_distance}\n")
                    print(f"{file_path} vs {hash_dict[existing_hashes]} calculated pHash distance: {phash_distance}")
                    print(f"{file_path} vs {hash_dict[existing_hashes]} calculated dHash distance: {dhash_distance}")
                if all_similar:
                    duplicate_path = os.path.join(duplicates_dir, f"{folder}_duplicate_{filename}")
                    shutil.move(file_path, duplicate_path)
                    print(f"Moved duplicate: {file_path} -> {duplicate_path}")
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            hash_dict[tuple(file_hashes)] = file_path

        # Step 4: Re-rename to ensure consistent numbering
        for extension in set(os.path.splitext(f)[1].lower() for f in renamed_files):
            highest_number = get_highest_existing_number(folder_path, folder, extension)
            # Filter out non-existent files before sorting
            valid_items = [(hashes, path) for hashes, path in hash_dict.items() if os.path.exists(path)]
            for file_hashes, file_path in sorted(valid_items, key=lambda x: os.path.getmtime(x[1])):
                current_ext = os.path.splitext(file_path)[1].lower()
                if current_ext != extension:
                    continue
                # Check if the file already has the correct name
                match = re.match(rf"{re.escape(folder)} \((\d+)\){re.escape(extension)}$", os.path.basename(file_path))
                if match and int(match.group(1)) >= highest_number:
                    print(f"Skipped re-renaming (already correct): {file_path}")
                    highest_number = max(highest_number, int(match.group(1)))
                    continue
                file_number = get_next_available_number(folder_path, folder, extension, highest_number)
                new_filename = f"{folder} ({file_number}){extension}"
                new_path = os.path.join(folder_path, new_filename)
                try:
                    if file_path != new_path:
                        os.rename(file_path, new_path)
                        print(f"Re-renamed after deduplication: {file_path} -> {new_path}")
                    highest_number = file_number
                except Exception as e:
                    print(f"Error re-renaming {file_path}: {e}")
                    move_to_corrupted(corrupted_dir, file_path, "re-renaming failure")

        print(f"Finished processing folder: {folder_path}")

def main():
    """Main function to initiate processing."""
    parser = argparse.ArgumentParser(description="Rename, clean, and deduplicate media files.")
    parser.add_argument("--phash-threshold", type=int, default=10, help="Hamming distance threshold for deduplication (phash, 16x16)")
    parser.add_argument("--dhash-threshold", type=int, default=2, help="Hamming distance threshold for deduplication (dhash, 16x16, images only)")
    parser.add_argument("--root-dir", type=str, default="Content", help="Root directory containing media files")
    args = parser.parse_args()

    root_directory = args.root_dir
    if not os.path.exists(root_directory):
        print(f"Directory '{root_directory}' not found.")
        return
    print(f"Processing media files in '{root_directory}' with phash_threshold={args.phash_threshold}, "
          f"dhash_threshold={args.dhash_threshold} (images), 7 (videos)...")
    process_folder(root_dir=args.root_dir, phash_threshold=args.phash_threshold, dhash_threshold=args.dhash_threshold)
    print("Processing complete.")

if __name__ == "__main__":
    main()