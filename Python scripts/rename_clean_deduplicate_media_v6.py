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

def remove_exif_image(image_path):
    try:
        image = Image.open(image_path)
        image.save(image_path, exif=b'')
        image.close()
        print(f"Removed EXIF data from: {image_path}")
    except Exception as e:
        print(f"Error removing EXIF from {image_path}: {e}")

def get_image_hash(image_path):
    try:
        with Image.open(image_path) as img:
            hash_value = imagehash.dhash(img)
            print(f"Hash for {image_path}: {str(hash_value)}")
            return hash_value
    except Exception as e:
        print(f"Error hashing {image_path}: {e}")
        return None

def get_video_frame_hashes(video_path, num_frames=3):
    temp_dir = None
    frame_paths = []
    try:
        # Check if ffmpeg is available
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise FileNotFoundError("FFmpeg not found in PATH. Please install FFmpeg and add it to PATH.")

        # Create temporary directory for frame extraction
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")
        temp_dirs.append(temp_dir)  # Track for global cleanup

        # Extract video duration using ffprobe
        duration_cmd = [ffmpeg_path, "-i", video_path]
        result = subprocess.run(duration_cmd, capture_output=True, text=True)
        duration = 10  # Default duration in seconds if ffprobe fails
        duration_pattern = r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})"
        for line in result.stderr.splitlines():
            match = re.search(duration_pattern, line)
            if match:
                h, m, s = map(float, match.groups())
                duration = h * 3600 + m * 60 + s
                break

        # Calculate frame extraction times
        interval = duration / (num_frames + 1)
        for i in range(1, num_frames + 1):
            output_path = os.path.join(temp_dir, f"frame_{i}.png")
            subprocess.run([
                ffmpeg_path, "-i", video_path, "-ss", str(i * interval),
                "-vframes", "1", "-q:v", "2", output_path
            ], check=True, capture_output=True, text=True)
            if os.path.exists(output_path):
                frame_paths.append(output_path)
                print(f"Created temporary frame: {output_path}")
            else:
                print(f"Failed to create temporary frame: {output_path}")

        # Compute hashes for extracted frames
        frame_hashes = []
        for frame_path in frame_paths:
            frame_hash = get_image_hash(frame_path)
            if frame_hash:
                frame_hashes.append(frame_hash)
        return frame_hashes if frame_hashes else None
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None
    finally:
        # Explicitly clean up temporary files and directory
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

def process_folder(root_dir):
    media_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.mp4', '.mov', '.avi'}
    duplicates_dir = os.path.join(root_dir, "Duplicates")
    os.makedirs(duplicates_dir, exist_ok=True)

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path) or folder == "Duplicates":
            continue

        print(f"\nProcessing folder: {folder}")
        media_files = [
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in media_extensions
        ]
        media_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

        # Remove EXIF data from images first
        for filename in media_files:
            file_path = os.path.join(folder_path, filename)
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith('image'):
                remove_exif_image(file_path)

        # Track hashes and processed files
        hash_dict = {}  # Store hash (image) or list of hashes (video): path
        file_counter = 1

        for filename in media_files:
            old_path = os.path.join(folder_path, filename)
            extension = os.path.splitext(filename)[1].lower()
            mime_type, _ = mimetypes.guess_type(old_path)

            # Compute hash based on file type
            file_hashes = None
            if mime_type and mime_type.startswith('image'):
                file_hashes = [get_image_hash(old_path)]
            elif mime_type and mime_type.startswith('video'):
                file_hashes = get_video_frame_hashes(old_path)
            if not file_hashes:
                continue

            # Check for duplicates
            is_duplicate = False
            for existing_hashes in hash_dict:
                if len(file_hashes) != len(existing_hashes):
                    continue
                # For images: single hash comparison; for videos: all frame hashes must be similar
                all_similar = True
                for new_hash, existing_hash in zip(file_hashes, existing_hashes):
                    if new_hash - existing_hash > 5:  # Hamming distance threshold
                        all_similar = False
                        break
                if all_similar:
                    duplicate_path = os.path.join(duplicates_dir, f"{folder}_duplicate_{os.path.basename(old_path)}")
                    shutil.move(old_path, duplicate_path)
                    print(f"Moved duplicate: {old_path} -> {duplicate_path}")
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            hash_dict[tuple(file_hashes)] = old_path

            # Rename file
            new_filename = f"{folder} ({file_counter}){extension}"
            new_path = os.path.join(folder_path, new_filename)
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
                file_counter += 1
            except Exception as e:
                print(f"Error processing {old_path}: {e}")

        # Log completion of subfolder processing
        print(f"Finished processing folder: {folder}")

def main():
    root_directory = "Content"
    if not os.path.exists(root_directory):
        print(f"Directory '{root_directory}' not found.")
        return
    print(f"Processing media files in '{root_directory}'...")
    process_folder(root_directory)
    print("Processing complete.")

if __name__ == "__main__":
    main()