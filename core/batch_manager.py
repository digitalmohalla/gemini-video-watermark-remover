"""
Batch Processing Manager.
Orchestrates multi-video queue execution (1 to 100s of videos),
handles destination path resolution, progress tracking, cancellation,
automatic retry logic, and ZIP packing.
"""

import os
import time
import uuid
import queue
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from .utils import (
    get_video_info,
    extract_thumbnail_base64,
    get_downloads_folder,
    create_zip_archive,
)
from .native_engine import NativeEngine
from .python_engine import PythonFrameEngine
from .profiler import IntelligentWatermarkProfiler


class BatchItem:
    """Represents a single video task in the queue."""

    def __init__(self, file_path: str):
        self.id = str(uuid.uuid4())[:8]
        self.input_path = str(Path(file_path).resolve())
        self.filename = Path(file_path).name

        # Extract metadata
        meta = get_video_info(self.input_path) or {}
        self.size_mb = meta.get("size_mb", 0.0)
        self.width = meta.get("width", 0)
        self.height = meta.get("height", 0)
        self.resolution = meta.get("resolution", "Unknown")
        self.duration = meta.get("duration", "Unknown")
        self.duration_sec = meta.get("duration_sec", 0.0)
        self.fps = meta.get("fps", 30.0)
        self.total_frames = meta.get("total_frames", 0)

        # Extract intelligent watermark & aspect ratio profile directly from media
        try:
            self.profile = IntelligentWatermarkProfiler().profile(self.input_path)
        except Exception:
            self.profile = None

        # Status tracking
        self.status = "pending"  # "pending", "processing", "completed", "error", "cancelled"
        self.progress_pct = 0.0
        self.current_frame = 0
        self.current_fps = 0.0
        self.status_msg = "Waiting in queue"
        self.error_message: Optional[str] = None
        self.output_path: Optional[str] = None
        self.elapsed_sec = 0.0
        self.thumbnail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "input_path": self.input_path,
            "filename": self.filename,
            "size_mb": self.size_mb,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "duration": self.duration,
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "status": self.status,
            "progress_pct": round(self.progress_pct, 1),
            "current_frame": self.current_frame,
            "current_fps": self.current_fps,
            "status_msg": self.status_msg,
            "error_message": self.error_message,
            "output_path": self.output_path,
            "elapsed_sec": self.elapsed_sec,
            "has_thumbnail": self.thumbnail is not None,
            "aspect_ratio": self.profile.aspect_ratio if self.profile else "16:9",
            "is_vertical": self.profile.is_vertical if self.profile else False,
            "is_square": self.profile.is_square if self.profile else False,
            "has_black_bars": self.profile.has_black_bars if self.profile else False,
            "content_aspect_ratio": self.profile.content_aspect_ratio if self.profile else "16:9",
            "watermark_found": self.profile.watermark_found if self.profile else False,
            "watermark_confidence": self.profile.confidence if self.profile else 0.0,
            "watermark_region": self.profile.region_str if self.profile else "br:auto",
            "watermark_size": f"{self.profile.width_wm}x{self.profile.height_wm}" if self.profile else "48x48",
            "measured_alpha": self.profile.measured_alpha if self.profile else 0.78,
        }


class BatchManager:
    """Coordinates batch processing across multiple videos."""

    def __init__(self):
        self.items: List[BatchItem] = []
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._is_paused = False

        # Engines
        self.native_engine: Optional[NativeEngine] = None
        try:
            self.native_engine = NativeEngine()
        except Exception as e:
            print(f"Native engine not initialized: {e}")

        self.python_engine = PythonFrameEngine()
        self._active_engine: Optional[Any] = None

        # Settings
        self.settings = {
            "engine": "python",              # "python" (Lossless reverse-alpha, recommended) or "native"
            "mark_type": "auto",             # "auto", "diamond", "veo"
            "denoise_mode": "off",           # "off" (pure reverse-alpha, lossless), "soft", "ai", "ns", "telea"
            "destination_type": "same_folder",  # "same_folder", "downloads", "custom"
            "custom_destination": "",
            "subfolder_name": "cleaned_videos",
            "use_subfolder": False,
            "output_suffix": "_clean",
            "auto_zip": False,
            "force_mode": False,
            "force_cpu": False,
            "veo_alpha": 0.0,
            "auto_retry": True,              # Automatically retry with variant locks on failure
        }

        # Subscribers for real-time progress events
        self._subscribers: List[queue.Queue] = []
        self.log_history: List[str] = []

    def subscribe(self) -> queue.Queue:
        """Subscribe to Server-Sent Event updates."""
        q = queue.Queue(maxsize=100)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        """Unsubscribe from updates."""
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast(self, event_type: str, data: Any):
        """Send event to all active SSE subscribers."""
        msg = {"type": event_type, "data": data, "timestamp": time.time()}
        with self._lock:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass

    def add_log(self, text: str):
        """Append to live algorithm log."""
        self.log_history.append(text)
        if len(self.log_history) > 1000:
            self.log_history.pop(0)
        self._broadcast("log", text)

    def add_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Add multiple video files to the processing queue."""
        added = []
        existing_paths = {item.input_path for item in self.items}

        for path_str in file_paths:
            p = Path(path_str).resolve()
            if not p.is_file():
                continue
            if str(p) in existing_paths:
                continue

            item = BatchItem(str(p))
            threading.Thread(
                target=self._load_item_thumbnail, args=(item,), daemon=True
            ).start()

            with self._lock:
                self.items.append(item)
                existing_paths.add(str(p))
                added.append(item.to_dict())

        self._broadcast("queue_updated", self.get_state())
        return added

    def _load_item_thumbnail(self, item: BatchItem):
        thumb = extract_thumbnail_base64(item.input_path)
        if thumb:
            item.thumbnail = thumb
            self._broadcast("thumbnail_ready", {"id": item.id, "thumbnail": thumb})

    def remove_item(self, item_id: str):
        """Remove an item from the queue."""
        with self._lock:
            self.items = [it for it in self.items if it.id != item_id]
        self._broadcast("queue_updated", self.get_state())

    def retry_item(self, item_id: str):
        """Retry a specific item in the queue."""
        with self._lock:
            for it in self.items:
                if it.id == item_id:
                    it.status = "pending"
                    it.progress_pct = 0.0
                    it.status_msg = "Queued for retry"
                    it.error_message = None
                    break
        self._broadcast("queue_updated", self.get_state())
        self.start_batch()

    def retry_failed(self):
        """Retry all failed items in the queue."""
        with self._lock:
            for it in self.items:
                if it.status == "error":
                    it.status = "pending"
                    it.progress_pct = 0.0
                    it.status_msg = "Queued for retry"
                    it.error_message = None
        self._broadcast("queue_updated", self.get_state())
        self.start_batch()

    def clear_queue(self, completed_only: bool = False):
        """Clear the queue."""
        with self._lock:
            if completed_only:
                self.items = [
                    it for it in self.items if it.status not in ("completed", "error", "cancelled")
                ]
            else:
                self.items = [it for it in self.items if it.status == "processing"]
        self._broadcast("queue_updated", self.get_state())

    def update_settings(self, new_settings: Dict[str, Any]):
        """Update batch processing settings."""
        with self._lock:
            self.settings.update(new_settings)
        self._broadcast("settings_updated", self.settings)

    def resolve_output_path(self, input_path: str) -> Path:
        """Calculate target output file path based on current settings."""
        in_p = Path(input_path)
        suffix = self.settings.get("output_suffix", "_clean")
        dest_type = self.settings.get("destination_type", "same_folder")
        use_subfolder = self.settings.get("use_subfolder", False)
        subfolder_name = self.settings.get("subfolder_name", "cleaned_videos").strip()

        if dest_type == "downloads":
            target_dir = get_downloads_folder()
        elif dest_type == "custom":
            custom_dir = self.settings.get("custom_destination", "").strip()
            target_dir = Path(custom_dir) if custom_dir else in_p.parent
        else:  # "same_folder"
            target_dir = in_p.parent

        if use_subfolder and subfolder_name:
            target_dir = target_dir / subfolder_name

        target_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{in_p.stem}{suffix}{in_p.suffix}"
        return target_dir / out_name

    def start_batch(self):
        """Start or resume processing pending items."""
        if self._worker_thread and self._worker_thread.is_alive():
            self._is_paused = False
            self._broadcast("batch_resumed", {})
            return

        self._stop_requested = False
        self._is_paused = False
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        self._broadcast("batch_started", {})

    def pause_batch(self):
        """Pause processing after current item finishes."""
        self._is_paused = True
        self._broadcast("batch_paused", {})

    def cancel_batch(self):
        """Cancel batch processing immediately."""
        self._stop_requested = True
        self._is_paused = False
        if self._active_engine:
            self._active_engine.cancel()
        self._broadcast("batch_cancelled", {})

    def _process_queue(self):
        """Worker thread processing items sequentially."""
        while not self._stop_requested:
            if self._is_paused:
                time.sleep(0.5)
                continue

            target_item: Optional[BatchItem] = None
            with self._lock:
                for it in self.items:
                    if it.status == "pending":
                        target_item = it
                        break

            if target_item is None:
                break

            self._process_single_item(target_item)

        if not self._stop_requested and self.settings.get("auto_zip", False):
            self.create_zip_from_completed()

        self._broadcast("batch_finished", self.get_state())

    def _process_single_item(self, item: BatchItem):
        """Execute watermark removal on one video with multi-tier retry fallback."""
        item.status = "processing"
        item.progress_pct = 0.0
        item.status_msg = "Starting engine..."
        self._broadcast("item_updated", item.to_dict())
        self._broadcast("queue_updated", self.get_state())

        out_path = self.resolve_output_path(item.input_path)
        item.output_path = str(out_path)

        def on_progress(pct, cur_frame, total, fps, msg):
            item.progress_pct = pct
            item.current_frame = cur_frame
            item.current_fps = fps
            item.status_msg = msg
            self._broadcast("item_progress", item.to_dict())

        def on_log(line):
            self.add_log(f"[{item.filename}] {line}")

        engine_type = self.settings.get("engine", "native")
        res = {"status": "error", "message": "Unknown engine"}

        profile = getattr(item, "profile", None)
        region_override = profile.region_str if (profile and profile.watermark_found) else None
        alpha_override = profile.measured_alpha if (profile and profile.watermark_found) else (self.settings.get("veo_alpha", 0.0) or None)
        mark_override = profile.mark_type if (profile and profile.watermark_found) else self.settings.get("mark_type", "auto")

        if profile and profile.watermark_found:
            on_log(f"[Profile] Aspect Ratio: {profile.aspect_ratio} ({profile.content_aspect_ratio} content). Watermark: {profile.width_wm}x{profile.height_wm} @ {profile.region_str} (Alpha: {int(profile.measured_alpha*100)}%)")
        elif profile and not profile.watermark_found and not self.settings.get("force_mode", False):
            on_log(f"[Profile] No watermark detected ({item.filename} is already clean). Preserving original video without modification.")
            import shutil
            shutil.copy2(item.input_path, str(out_path))
            item.status = "completed"
            item.progress_pct = 100.0
            item.status_msg = "Clean (No Watermark Detected)"
            self.save_queue()
            return

        from .utils import is_image
        if is_image(item.input_path):
            # Process as Static Image (Gemini Diamond Reverse-Alpha)
            if engine_type == "native" and self.native_engine:
                self._active_engine = self.native_engine
                res = self.native_engine.process_image(
                    input_path=item.input_path,
                    output_path=str(out_path),
                    region=region_override,
                    denoise_mode=self.settings.get("denoise_mode", "off"),
                    force_mode=True if region_override else self.settings.get("force_mode", False),
                    progress_callback=on_progress,
                    log_callback=on_log,
                )
                # Auto-fallback to Python engine for images if native skipped/failed
                if res.get("status") != "success" and not self._stop_requested and self.settings.get("auto_retry", True):
                    on_log(f"[Fallback] Native skipped ({res.get('message')}). Launching Python Reverse-Alpha Engine...")
                    on_progress(50.0, 0, 1, 0.0, "Auto-Fallback: Using Python Engine...")
                    self._active_engine = self.python_engine
                    res = self.python_engine.process_image(
                        input_path=item.input_path,
                        output_path=str(out_path),
                        region=region_override,
                        progress_callback=on_progress,
                        log_callback=on_log,
                    )
            else:
                self._active_engine = self.python_engine
                res = self.python_engine.process_image(
                    input_path=item.input_path,
                    output_path=str(out_path),
                    region=region_override,
                    progress_callback=on_progress,
                    log_callback=on_log,
                )
        elif engine_type == "native" and self.native_engine:
            self._active_engine = self.native_engine
            # If explicit region was found dynamically by profiler or black bars exist, pass it to lock 100% confidence
            video_region = None
            if region_override:
                video_region = region_override
            elif profile and profile.watermark_found and profile.region_str:
                video_region = profile.region_str
            elif profile and profile.has_black_bars and profile.region_str:
                video_region = profile.region_str
            res = self.native_engine.process_video(
                input_path=item.input_path,
                output_path=str(out_path),
                mark_type=mark_override,
                denoise_mode=self.settings.get("denoise_mode", "off"),
                force_mode=True if video_region else self.settings.get("force_mode", False),
                force_cpu=self.settings.get("force_cpu", False),
                region=video_region,
                veo_alpha=self.settings.get("veo_alpha", 0.0) or None,
                video_width=item.width,
                video_height=item.height,
                auto_retry=self.settings.get("auto_retry", True),
                progress_callback=on_progress,
                log_callback=on_log,
            )

            # Auto-fallback to Python engine if native still failed
            if res.get("status") != "success" and not self._stop_requested and self.settings.get("auto_retry", True):
                on_log(f"[Fallback] Native detection missed ({res.get('message')}). Launching Python Frame Engine...")
                on_progress(5.0, 0, item.total_frames, 0.0, "Auto-Fallback: Using Python Engine...")
                self._active_engine = self.python_engine
                res = self.python_engine.process_video(
                    input_path=item.input_path,
                    output_path=str(out_path),
                    region=video_region,
                    progress_callback=on_progress,
                    log_callback=on_log,
                )

        else:
            self._active_engine = self.python_engine
            res = self.python_engine.process_video(
                input_path=item.input_path,
                output_path=str(out_path),
                region=video_region if 'video_region' in locals() and video_region else region_override,
                progress_callback=on_progress,
                log_callback=on_log,
            )

        self._active_engine = None

        if res.get("status") == "success":
            item.status = "completed"
            item.progress_pct = 100.0
            item.elapsed_sec = res.get("elapsed_sec", 0.0)
            item.current_fps = res.get("fps", 0.0)
            item.status_msg = f"Done in {item.elapsed_sec}s ({item.current_fps} fps)"
        elif res.get("status") == "cancelled":
            item.status = "cancelled"
            item.status_msg = "Cancelled"
        else:
            item.status = "error"
            item.error_message = res.get("message", "Processing failed")
            item.status_msg = f"Error: {item.error_message}"

        self._broadcast("item_updated", item.to_dict())
        self._broadcast("queue_updated", self.get_state())

    def create_zip_from_completed(self) -> Optional[str]:
        """Bundle all completed videos into a ZIP archive."""
        completed_paths = [
            it.output_path
            for it in self.items
            if it.status == "completed" and it.output_path and Path(it.output_path).exists()
        ]

        if not completed_paths:
            return None

        dest_type = self.settings.get("destination_type", "same_folder")
        if dest_type == "downloads":
            base_dir = get_downloads_folder()
        elif dest_type == "custom" and self.settings.get("custom_destination"):
            base_dir = Path(self.settings.get("custom_destination"))
        else:
            base_dir = Path(completed_paths[0]).parent

        zip_name = f"cleaned_videos_{int(time.time())}.zip"
        zip_path = base_dir / zip_name

        self.add_log(f"Creating ZIP archive: {zip_name} with {len(completed_paths)} videos...")
        created_path = create_zip_archive(completed_paths, str(zip_path))
        self.add_log(f"ZIP created successfully: {created_path}")
        self._broadcast("zip_created", {"zip_path": created_path, "filename": zip_name})
        return created_path

    def get_state(self) -> Dict[str, Any]:
        """Return full snapshot of the queue and batch state."""
        with self._lock:
            items_dict = [it.to_dict() for it in self.items]
            total = len(self.items)
            completed = sum(1 for it in self.items if it.status == "completed")
            processing = sum(1 for it in self.items if it.status == "processing")
            failed = sum(1 for it in self.items if it.status == "error")
            pending = sum(1 for it in self.items if it.status == "pending")

            overall_pct = 0.0
            if total > 0:
                completed_sum = completed * 100.0
                active_sum = sum(it.progress_pct for it in self.items if it.status == "processing")
                overall_pct = round((completed_sum + active_sum) / total, 1)

            is_running = self._worker_thread is not None and self._worker_thread.is_alive()

            return {
                "items": items_dict,
                "total_count": total,
                "completed_count": completed,
                "processing_count": processing,
                "failed_count": failed,
                "pending_count": pending,
                "overall_progress_pct": overall_pct,
                "is_running": is_running,
                "is_paused": self._is_paused,
                "settings": self.settings,
            }
