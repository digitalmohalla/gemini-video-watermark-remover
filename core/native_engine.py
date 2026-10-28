"""
Native C++ Reverse-Alpha GPU Engine Wrapper for Video Watermark Removal.
Drives GeminiWatermarkTool-Video.exe with real-time progress parsing,
automatic frame probing, adaptive alpha scaling, Vulkan GPU acceleration,
and intelligent auto-retry fallback when standard auto-detection misses low-contrast frames.
"""

import os
import re
import sys
import time
import subprocess
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

import cv2
import numpy as np

from .utils import get_native_exe_path, detect_content_bbox


class NativeEngine:
    """Wrapper for the native C++ video watermark removal tool."""

    def __init__(self, exe_path: Optional[str] = None):
        self.exe_path = Path(exe_path) if exe_path else get_native_exe_path()
        if not self.exe_path.exists():
            raise FileNotFoundError(f"Native binary not found at: {self.exe_path}")

        self._current_proc: Optional[subprocess.Popen] = None
        self._cancelled = False

    def cancel(self):
        """Cancel current running video processing."""
        self._cancelled = True
        if self._current_proc:
            try:
                self._current_proc.terminate()
                self._current_proc.kill()
            except Exception:
                pass

    def _execute_cli(
        self,
        cmd: List[str],
        start_time: float,
        out_p: Path,
        progress_callback: Optional[Callable[[float, int, int, float, str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the native binary and monitor progress."""
        current_frame = 0
        total_frames = 0
        current_fps = 0.0
        progress_pct = 0.0
        details = ""
        output_lines = []

        pct_regex = re.compile(r"(\d+)%\s*\((\d+)/(\d+)\)")
        fps_regex = re.compile(r"\((\d+\.?\d*)\s*fps\)")

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        self._current_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
            encoding="utf-8",
            errors="replace",
        )

        for raw_line in iter(self._current_proc.stdout.readline, ""):
            if self._cancelled:
                break

            line = raw_line.strip()
            if not line:
                continue

            output_lines.append(line)
            if len(output_lines) > 50:
                output_lines.pop(0)

            if log_callback:
                log_callback(line)

            m_pct = pct_regex.search(line)
            if m_pct:
                progress_pct = float(m_pct.group(1))
                current_frame = int(m_pct.group(2))
                total_frames = int(m_pct.group(3))

                elapsed = time.time() - start_time
                if elapsed > 0 and current_frame > 0:
                    current_fps = round(current_frame / elapsed, 1)

                if progress_callback:
                    progress_callback(
                        progress_pct,
                        current_frame,
                        total_frames,
                        current_fps,
                        f"Processing frame {current_frame}/{total_frames} ({progress_pct:.0f}%)",
                    )

            if "Smart search located" in line or "Locked region" in line:
                details = line

            m_fps = fps_regex.search(line)
            if m_fps:
                current_fps = float(m_fps.group(1))

        self._current_proc.stdout.close()
        return_code = self._current_proc.wait()

        if self._cancelled:
            if out_p.exists():
                try:
                    out_p.unlink()
                except Exception:
                    pass
            return {"status": "cancelled", "message": "Processing cancelled by user"}

        total_time = round(time.time() - start_time, 2)
        success = (return_code == 0) and out_p.exists() and (out_p.stat().st_size > 0)

        # Extract meaningful skip or error reason
        error_msg = f"Native engine exited with code {return_code}"
        has_heavy_skip = False
        skip_regex = re.compile(r'(\d+)/(\d+)\s+frames\s+\((\d+)\s+skip\)')
        for line in reversed(output_lines):
            m_s = skip_regex.search(line)
            if m_s:
                total_f_count = int(m_s.group(2))
                skip_count = int(m_s.group(3))
                if skip_count > max(2, int(total_f_count * 0.10)):
                    has_heavy_skip = True
                    error_msg = f"Watermark unblending skipped {skip_count}/{total_f_count} frames"
                break
            if "[SKIP]" in line or "no Gemini" in line or "error" in line.lower():
                error_msg = line
                break

        if has_heavy_skip:
            success = False

        return {
            "status": "success" if success else "error",
            "return_code": return_code,
            "message": error_msg,
            "output_path": str(out_p),
            "elapsed_sec": total_time,
            "fps": current_fps,
            "total_frames": total_frames,
            "details": details,
            "output_lines": output_lines,
        }

    def process_video(
        self,
        input_path: str,
        output_path: str,
        mark_type: str = "auto",       # "auto", "diamond", "veo"
        denoise_mode: str = "off",     # "off" (recommended pure reverse-alpha), "ai", "soft", "ns", "telea"
        force_mode: bool = False,
        region: Optional[str] = None,  # e.g. "1179,642,20,20" or "br:mx,my,w,h"
        threshold: float = 0.25,
        veo_alpha: Optional[float] = None,
        force_cpu: bool = False,
        variant: Optional[str] = None, # "720p-1", "720p-2", etc.
        video_width: int = 0,
        video_height: int = 0,
        auto_retry: bool = True,
        progress_callback: Optional[Callable[[float, int, int, float, str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single video using the native C++ reverse alpha engine.
        Includes automatic multi-variant retries if auto-detection misses the watermark.
        """
        self._cancelled = False
        in_p = Path(input_path).resolve()
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not in_p.exists():
            return {"status": "error", "message": f"Input file not found: {in_p}"}

        # Attempt 1: Standard configured run
        def build_cmd(cur_variant=None, cur_threshold=threshold, cur_legacy=False, cur_region=None):
            c = [str(self.exe_path), "-i", str(in_p), "-o", str(out_p), "--veo"]
            if mark_type in ("auto", "diamond", "veo"):
                c.extend(["--mark", mark_type])
            if denoise_mode in ("ai", "off", "soft", "ns", "telea"):
                c.extend(["--denoise", denoise_mode])
            else:
                c.extend(["--denoise", "off"])
            effective_region = cur_region or region
            if force_mode or effective_region:
                c.append("-f")
            if effective_region and effective_region.strip():
                c.extend(["--region", effective_region.strip()])
            else:
                # Enable intelligent snap search for letterbox/pillarbox/custom aspect ratios
                c.extend(["--fallback-region", "br:auto", "--snap", "--snap-threshold", "0.20"])
            if cur_threshold != 0.25:
                c.extend(["-t", str(cur_threshold)])
            if veo_alpha is not None and veo_alpha > 0:
                c.extend(["--veo-alpha", str(veo_alpha)])
            if force_cpu:
                c.append("--cpu")
            if cur_variant:
                c.extend(["--variant", cur_variant])
            if cur_legacy:
                c.append("--legacy")
            return c

        start_time = time.time()
        cmd = build_cmd(cur_variant=variant)
        res = self._execute_cli(cmd, start_time, out_p, progress_callback, log_callback)

        if res["status"] == "success" or res.get("status") == "cancelled" or not auto_retry:
            return res

        # -------------------------------------------------------------
        # Auto-Retry Strategy: Handle "no diamond detected" or skipped frames
        # -------------------------------------------------------------
        is_720p = (video_width == 1280 and video_height == 720) or (video_width == 720 and video_height == 1280) or ("1280x720" in res.get("message", ""))
        is_1080p = (video_width == 1920 and video_height == 1080) or (video_width == 1080 and video_height == 1920) or ("1920x1080" in res.get("message", ""))

        fallback_strategies = []
        if is_720p:
            # 1. Standard 720p diamond coordinates: (1136, 576, 48, 48) - 100% confidence lock
            fallback_strategies.append((None, 0.25, False, "1136,576,48,48", "Retrying with standard 720p diamond lock (1136,576)..."))
            fallback_strategies.append(("720p-1", 0.25, False, None, "Retrying with 720p-1 diamond profile..."))
            fallback_strategies.append(("720p-2", 0.25, False, None, "Retrying with 720p-2 compact profile..."))
            fallback_strategies.append((None, 0.25, True, None, "Retrying with legacy Veo text profile..."))
        elif is_1080p:
            fallback_strategies.append((None, 0.25, False, "1728,888,96,96", "Retrying with standard 1080p diamond lock..."))
            fallback_strategies.append((None, 0.10, False, None, "Retrying with relaxed sensitivity threshold..."))
            fallback_strategies.append((None, 0.25, True, None, "Retrying with legacy Veo text profile..."))
        else:
            fallback_strategies.append((None, 0.10, False, None, "Retrying with relaxed sensitivity threshold..."))
            fallback_strategies.append((None, 0.25, True, None, "Retrying with legacy Veo text profile..."))

        for fb_variant, fb_thresh, fb_legacy, fb_reg, fb_msg in fallback_strategies:
            if self._cancelled:
                break

            if log_callback:
                log_callback(f"[Auto-Retry] {fb_msg}")
            if progress_callback:
                progress_callback(5.0, 0, 0, 0.0, fb_msg)

            time.sleep(0.3)
            retry_cmd = build_cmd(cur_variant=fb_variant, cur_threshold=fb_thresh, cur_legacy=fb_legacy, cur_region=fb_reg)
            retry_res = self._execute_cli(retry_cmd, time.time(), out_p, progress_callback, log_callback)

            if retry_res["status"] == "success":
                retry_res["details"] += f" (Resolved via {fb_msg})"
                return retry_res

        # Return the descriptive error
        return res

    def process_image(
        self,
        input_path: str,
        output_path: str,
        region: Optional[str] = None,
        denoise_mode: str = "off",
        force_mode: bool = False,
        progress_callback: Optional[Callable[[float, int, int, float, str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single image using the native C++ reverse alpha engine.
        Automatically handles black padding/borders and uses snap detection for sub-pixel accuracy.
        """
        self._cancelled = False
        in_p = Path(input_path).resolve()
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not in_p.exists():
            return {"status": "error", "message": f"Input file not found: {in_p}"}

        start_time = time.time()

        # If an explicit region was discovered from frame profiling, run directly
        if region and region.strip():
            cmd_exact = [
                str(self.exe_path),
                "-i", str(in_p),
                "-o", str(out_p),
                "-f",
                "--region", region.strip(),
                "--denoise", denoise_mode if denoise_mode in ("ai", "off", "soft", "ns", "telea") else "off",
            ]
            res_exact = self._execute_cli(cmd_exact, start_time, out_p, progress_callback, log_callback)
            if res_exact.get("status") == "success":
                return res_exact
        im = cv2.imread(str(in_p))
        if im is None:
            return {"status": "error", "message": "Failed to read image with OpenCV"}

        h, w = im.shape[:2]
        x1, y1, x2, y2 = detect_content_bbox(im)
        has_bars = (x1 > 0 or y1 > 0 or x2 < w or y2 < h)

        temp_crop_in: Optional[Path] = None
        temp_crop_out: Optional[Path] = None
        target_in = in_p
        target_out = out_p

        try:
            if has_bars:
                content = im[y1:y2, x1:x2]
                temp_crop_in = in_p.parent / f"_tmp_in_{in_p.stem}.png"
                temp_crop_out = in_p.parent / f"_tmp_out_{in_p.stem}.png"
                cv2.imwrite(str(temp_crop_in), content)
                target_in = temp_crop_in
                target_out = temp_crop_out

            # Build base command with snap search
            cmd = [
                str(self.exe_path),
                "-i", str(target_in),
                "-o", str(target_out),
                "--fallback-region", "br:auto",
                "--snap",
                "--snap-threshold", "0.20",
                "--denoise", denoise_mode if denoise_mode in ("ai", "off", "soft", "ns", "telea") else "off",
            ]
            if force_mode:
                cmd.append("-f")

            res = self._execute_cli(cmd, start_time, target_out, progress_callback, log_callback)

            # If skipped or failed, retry with force mode
            if res.get("status") != "success" and not force_mode and not self._cancelled:
                if log_callback:
                    log_callback("[Image] Retrying with forced reverse-alpha unblending...")
                cmd_retry = [
                    str(self.exe_path),
                    "-i", str(target_in),
                    "-o", str(target_out),
                    "-f",
                    "--fallback-region", "br:auto",
                    "--denoise", denoise_mode if denoise_mode in ("ai", "off", "soft", "ns", "telea") else "off",
                ]
                res = self._execute_cli(cmd_retry, start_time, target_out, progress_callback, log_callback)

            if res.get("status") == "success" and has_bars:
                processed_content = cv2.imread(str(target_out))
                if processed_content is not None:
                    im[y1:y2, x1:x2] = processed_content
                    cv2.imwrite(str(out_p), im)
                    res["output_path"] = str(out_p)

            return res

        finally:
            # Clean up temporary files
            if temp_crop_in and temp_crop_in.exists():
                try:
                    temp_crop_in.unlink()
                except Exception:
                    pass
            if temp_crop_out and temp_crop_out.exists():
                try:
                    temp_crop_out.unlink()
                except Exception:
                    pass
