#!/usr/bin/env python3
"""
Setup & Dependency Downloader for Gemini & Veo Video Watermark Remover.
Automatically installs Python requirements and downloads required binaries
(GeminiWatermarkTool-Video.exe and FFmpeg) if they are missing.
"""

import os
import sys
import io
import shutil
import zipfile
import subprocess
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BIN_DIR = BASE_DIR / "bin"
BIN_DIR.mkdir(parents=True, exist_ok=True)

NATIVE_EXE = BIN_DIR / "GeminiWatermarkTool-Video.exe"
FFMPEG_EXE = BIN_DIR / "ffmpeg.exe"

NATIVE_TOOL_URL = "https://github.com/allenk/VeoWatermarkRemover/releases/download/v0.6.5-demo/GeminiWatermarkTool-Windows-x64-Video.zip"


def install_requirements():
    """Install required python packages via pip."""
    print("[1/3] Checking & installing Python requirements...")
    req_file = BASE_DIR / "requirements.txt"
    if req_file.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    print("      Requirements satisfied.\n")


def ensure_native_engine():
    """Download bundled native C++ reverse-alpha engine if missing."""
    print("[2/3] Checking native C++ GPU watermark engine...")
    if NATIVE_EXE.exists() and NATIVE_EXE.stat().st_size > 10000000:
        print(f"      Engine binary already present: {NATIVE_EXE.name} ({round(NATIVE_EXE.stat().st_size / 1024 / 1024, 1)} MB)")
        return

    print("      Downloading GeminiWatermarkTool-Video binary from release...")
    req = urllib.request.Request(NATIVE_TOOL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()

    print(f"      Downloaded {round(len(data) / 1024 / 1024, 1)} MB. Extracting to bin/...")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(str(BIN_DIR))

    if NATIVE_EXE.exists():
        print("      Native engine installed successfully!\n")
    else:
        print("      Warning: Native engine file was not found in extracted archive.\n")


def ensure_ffmpeg():
    """Ensure static ffmpeg binary is present in bin/."""
    print("[3/3] Checking static FFmpeg binary...")
    if FFMPEG_EXE.exists() and FFMPEG_EXE.stat().st_size > 10000000:
        print(f"      FFmpeg binary already present: {FFMPEG_EXE.name}")
        return

    try:
        import imageio_ffmpeg
        ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(ffmpeg_src):
            shutil.copy(ffmpeg_src, str(FFMPEG_EXE))
            print("      FFmpeg copied to bin/ successfully!\n")
            return
    except Exception:
        pass

    print("      Installing imageio-ffmpeg to source static binary...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
    import imageio_ffmpeg
    ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    shutil.copy(ffmpeg_src, str(FFMPEG_EXE))
    print("      FFmpeg installed to bin/ successfully!\n")


def main():
    print("=" * 65)
    print("  GEMINI & VEO VIDEO WATERMARK REMOVER - SETUP")
    print("=" * 65 + "\n")
    install_requirements()
    ensure_native_engine()
    ensure_ffmpeg()
    print("=" * 65)
    print("  SETUP COMPLETE! You can now run the app via:")
    print("    - run.bat         (Desktop Application)")
    print("    - run_browser.bat (Web Browser Interface)")
    print("=" * 65)


if __name__ == "__main__":
    main()
