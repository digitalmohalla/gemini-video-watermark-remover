"""
Utility functions for Gemini Watermark Remover:
- System hardware detection (GPU/CPU/RAM)
- Video metadata & thumbnail extraction
- Windows native file/folder pickers
- File Explorer integration
- Zip archive creation
"""

import os
import sys
import json
import zipfile
import subprocess
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import cv2
import numpy as np


def get_base_dir() -> Path:
    """Return the application base directory."""
    return Path(__file__).resolve().parent.parent


def get_bin_dir() -> Path:
    """Return the bundled bin directory containing native tools."""
    return get_base_dir() / "bin"


def get_native_exe_path() -> Path:
    """Return path to the bundled native video watermark removal executable."""
    return get_bin_dir() / "GeminiWatermarkTool-Video.exe"


def get_ffmpeg_path() -> Path:
    """Return path to bundled ffmpeg executable."""
    return get_bin_dir() / "ffmpeg.exe"


def get_downloads_folder() -> Path:
    """Return the user's default Downloads directory on Windows."""
    # Try userprofile / Downloads
    home = Path.home()
    downloads = home / "Downloads"
    if downloads.exists():
        return downloads

    # Fallback to home
    return home


def get_system_info() -> Dict[str, Any]:
    """Retrieve system hardware details (CPU, RAM, GPU)."""
    info = {
        "os": "Windows",
        "cpu_count": os.cpu_count() or 4,
        "gpus": [],
        "has_nvidia": False,
        "native_engine_ready": get_native_exe_path().exists(),
        "ffmpeg_ready": get_ffmpeg_path().exists(),
    }

    try:
        # Detect GPUs using PowerShell Get-CimInstance
        ps_cmd = (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object -Property Name, DriverVersion, AdapterRAM | "
            "ConvertTo-Json"
        )
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                name = item.get("Name", "Unknown GPU")
                info["gpus"].append(name)
                if "NVIDIA" in name.upper() or "GEFORCE" in name.upper():
                    info["has_nvidia"] = True
    except Exception:
        pass

    if not info["gpus"]:
        info["gpus"] = ["DirectX / Vulkan GPU"]

    return info


def get_video_info(video_path: str) -> Optional[Dict[str, Any]]:
    """Extract metadata for a given video file."""
    p = Path(video_path)
    if not p.is_file():
        return None

    file_size_mb = round(p.stat().st_size / (1024 * 1024), 2)

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return {
            "path": str(p),
            "filename": p.name,
            "size_mb": file_size_mb,
            "width": 0,
            "height": 0,
            "fps": 0,
            "total_frames": 0,
            "duration": "Unknown",
            "duration_sec": 0,
        }

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_sec = round(total_frames / fps, 1) if fps > 0 else 0
    mins = int(duration_sec // 60)
    secs = int(duration_sec % 60)
    duration_str = f"{mins:02d}:{secs:02d}"

    return {
        "path": str(p),
        "filename": p.name,
        "size_mb": file_size_mb,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration": duration_str,
        "duration_sec": duration_sec,
    }


def extract_thumbnail_base64(video_path: str, max_size: int = 160) -> Optional[str]:
    """Extract a thumbnail from the video and return as base64 JPEG string."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Seek to frame ~15% into video to avoid dark opening frames
        target_frame = max(1, min(int(total_frames * 0.15), total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None

        # Resize keeping aspect ratio
        h, w = frame.shape[:2]
        if w > h:
            new_w = max_size
            new_h = int(h * (max_size / w))
        else:
            new_h = max_size
            new_w = int(w * (max_size / h))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        import base64
        return base64.b64encode(buffer).decode("utf-8")
    except Exception:
        return None


def show_native_open_file_dialog() -> List[str]:
    """
    Launch Windows native Open File dialog supporting multi-selection.
    Runs via python subprocess or tkinter in separate thread.
    """
    script = """
import sys, json, tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
filetypes = [
    ("Video Files", "*.mp4 *.mov *.webm *.mkv *.avi *.m4v *.flv"),
    ("All Files", "*.*")
]
files = filedialog.askopenfilenames(
    title="Select Videos to Remove Watermark (Multiple Supported)",
    filetypes=filetypes
)
print(json.dumps(list(files)))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            files = json.loads(proc.stdout.strip())
            return files
    except Exception as e:
        print(f"Error opening file dialog: {e}")
    return []


def show_native_folder_dialog(title: str = "Select Destination Folder") -> Optional[str]:
    """Launch Windows native Folder picker dialog."""
    script = f"""
import sys, json, tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
folder = filedialog.askdirectory(title="{title}")
print(json.dumps(folder))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            folder = json.loads(proc.stdout.strip())
            return folder if folder else None
    except Exception as e:
        print(f"Error opening folder dialog: {e}")
    return None


def open_in_explorer(target_path: str):
    """Open Windows File Explorer focused on the specified file or folder."""
    p = Path(target_path).resolve()
    if p.is_file():
        subprocess.Popen(f'explorer.exe /select,"{str(p)}"', shell=True)
    elif p.is_dir():
        subprocess.Popen(f'explorer.exe "{str(p)}"', shell=True)


def create_zip_archive(file_paths: List[str], zip_output_path: str, progress_callback=None) -> str:
    """Create a ZIP archive containing all given files."""
    out_p = Path(zip_output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    total = len(file_paths)
    with zipfile.ZipFile(out_p, "w", zipfile.ZIP_DEFLATED) as zipf:
        for idx, f in enumerate(file_paths):
            fp = Path(f)
            if fp.exists():
                zipf.write(fp, arcname=fp.name)
            if progress_callback:
                progress_callback(idx + 1, total)

    return str(out_p)
