# Gemini & Veo Video Watermark Remover 🎬✨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Platform: Windows](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows&logoColor=white)](https://microsoft.com)
[![Acceleration: Vulkan GPU](https://img.shields.io/badge/Acceleration-Vulkan%20GPU-red?logo=vulkan&logoColor=white)](https://www.vulkan.org/)
[![UI: Modern Desktop & Web](https://img.shields.io/badge/UI-WebView2%20%7C%20Browser-6366f1)](#-quick-start)

A high-performance, **100% offline open-source desktop application** designed to remove **Google Gemini** and **Google Veo** video watermarks with mathematical precision.

Runs completely on your local device with **zero server uploads**, preserving 100% privacy and uncompressed video/audio quality.

---

## ⚡ Why `geminiwatermarkremover.io` Failed on Video

Online utilities like `geminiwatermarkremover.io` rely on in-browser HTML5 `<canvas>` and JavaScript `MediaRecorder`:
1. **Severe Browser Bottlenecks**: Processing video frames inside a browser canvas drops frame rates to single digits (2–5 FPS) and drops frames.
2. **Audio Degradation & Desync**: In-browser video re-encoding compresses audio to low-bitrate Opus/AAC or strips stereo/multi-channel tracks entirely.
3. **Flashing & Missed Frames**: Static single-frame thresholding misses dark frames, motion blur, and opening fade-ins, causing the watermark removal to flicker on video.
4. **Different Video Layouts**: Google Veo and Gemini videos have dynamic aspect ratios, downscaled diamond variants (e.g. 48×48, 44×44, 20×20), and text wordmarks that require **multi-frame probing** and calibrated alpha profiles.

---

## 🚀 Key Features

* **Dual-Engine Architecture**:
  * **Native C++ GPU Engine (Ultra-Fast 40–120+ FPS, Recommended)**: Driven by Allen Kuo's native reverse-alpha engine with Vulkan GPU acceleration (NVIDIA/Intel/AMD), multi-frame probe alignment, adaptive alpha scaling, and Vulkan AI edge denoiser.
  * **Python Frame Extraction Pipeline**: Directly extracts each frame, applies calibrated reverse-alpha blending with temporal continuity, and streams frames into H.264 MP4 with lossless audio passthrough via FFmpeg.
* **Automatic Multi-Tier Retries**:
  * Low-contrast or bright scenes that fail standard auto-detection automatically retry with variant locks (`720p-1`, `720p-2`, `legacy`) and fallback to Python frame unblending.
* **Batch Processing (1 to 100s of Videos)**:
  * Select multiple videos at once using the **Windows Native File Picker** dialog.
  * Scan and add entire directories with the **Folder Picker**.
  * Seamless Drag & Drop directly into the application.
* **Flexible Destination Options**:
  * Save in the same folder as the original video (`_clean.mp4`).
  * Save in your Windows **Downloads** directory.
  * Save in a custom user-selected folder.
  * Group outputs into a subfolder (e.g., `cleaned_videos/`).
  * **Auto-create ZIP archive**: Automatically bundles all processed videos into a single downloadable ZIP file.
* **Audio & Quality Preservation**:
  * Direct audio passthrough (`-c:a copy`): original audio bitstreams are copied without re-encoding.
  * Visually lossless video output without blur or AI hallucination.
* **Real-time Status & Live Logs**:
  * Real-time progress bar, frame counter (`Frame 45/90`), and FPS per video.
  * Overall batch progress indicator.
  * Live algorithm activity console showing NCC template correlation, detected coordinates, and alpha scaling.

---

## 🛠️ How It Works (Reverse Alpha Blending)

Google overlays the semi-transparent watermark using standard linear alpha compositing:

$$\text{Watermarked} = \alpha \times \text{Logo} + (1 - \alpha) \times \text{Original}$$

Unlike destructive AI inpainting (which "guesses" or hallucinates missing pixels and blurs textures), this application mathematically solves for the original image:

$$\text{Original} = \frac{\text{Watermarked} - (\alpha \times \text{Logo})}{1 - \alpha}$$

By calibrating the exact alpha transparency map $\alpha$ and probing multiple frames across the video, the original pixel values are cleanly recovered with zero blur.

---

## 💻 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/digitalmohalla/gemini-video-watermark-remover.git
cd gemini-video-watermark-remover
```

### 2. Launch (Auto-installs dependencies on first run!)

#### Option A: Native Desktop App (Recommended)
Double-click:
```bat
run.bat
```
*This launches the application and closes the command prompt window immediately.*

#### Option B: Web Browser Mode
Double-click:
```bat
run_browser.bat
```
*Opens the interface in your default web browser.*

#### Option C: Console Debug Mode
Double-click:
```bat
run_console.bat
```
*Keeps the terminal window open to display live stdout/stderr debug logs.*

---

## 📁 Project Structure

```
gemini-video-watermark-remover/
├── bin/                            # Downloaded automatically by setup.py on first run
│   ├── GeminiWatermarkTool-Video.exe   # Native C++ GPU reverse-alpha engine
│   └── ffmpeg.exe                      # Static FFmpeg binary with H.264 & NVENC
├── core/
│   ├── assets/                         # Calibrated alpha maps (bg_48.png, bg_96.png)
│   ├── batch_manager.py                # Multi-video queue & worker coordinator
│   ├── native_engine.py                # Wrapper for C++ GPU Vulkan engine
│   ├── python_engine.py                # Frame extraction & FFmpeg audio stream muxer
│   └── utils.py                        # System detection, Windows dialogs, ZIP packer
├── server/
│   └── app.py                          # Flask REST API & SSE real-time streaming
├── ui/
│   ├── css/style.css                   # Modern dark-mode styling
│   ├── js/app.js                       # Frontend state & real-time updates
│   └── index.html                      # Main dashboard
├── app.py                              # Main application entry point
├── setup.py                            # One-time automated environment setup
├── setup.bat                           # Setup batch script
├── run.bat                             # Windows Desktop Launcher
├── run_browser.bat                     # Windows Browser Launcher
├── run_console.bat                     # Terminal Debug Launcher
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
└── README.md                           # Documentation
```

---

## 🙏 Credits & Acknowledgments
* **Allen Kuo (`allenk`)**: For the pioneering research and native C++ reverse alpha blending engine in [GeminiWatermarkTool](https://github.com/allenk/GeminiWatermarkTool) and [VeoWatermarkRemover](https://github.com/allenk/VeoWatermarkRemover).
* **Journey-ad & Community**: For the foundational alpha template research.
* **FFmpeg**: For the legendary multimedia framework enabling lossless audio stream passthrough.

---

## 🔒 100% Offline & Private
All video processing, frame extraction, reverse-alpha calculations, and video encoding run entirely locally on your Windows machine. No network traffic or cloud servers are used.
