"""
Flask Web Server and REST API for Gemini Watermark Remover.
Provides endpoints for UI rendering, Windows native dialogs, SSE streaming,
batch queue control, and file explorer integration.
"""

import os
import sys
import json
import base64
from pathlib import Path
from typing import Dict, Any

from flask import (
    Flask,
    request,
    jsonify,
    Response,
    send_file,
    send_from_directory,
    stream_with_context,
)

from core.utils import (
    get_system_info,
    get_downloads_folder,
    open_in_explorer,
    show_native_open_file_dialog,
    show_native_folder_dialog,
)
from core.batch_manager import BatchManager


def create_app(batch_manager: BatchManager) -> Flask:
    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    app = Flask(__name__, static_folder=str(ui_dir), static_url_path="")

    # Store reference to manager
    app.batch_manager = batch_manager
    latest_zip_path = {"path": None}

    @app.route("/")
    def index():
        return send_from_directory(str(ui_dir), "index.html")

    @app.route("/api/system-info", methods=["GET"])
    def system_info():
        info = get_system_info()
        info["downloads_folder"] = str(get_downloads_folder())
        return jsonify(info)

    @app.route("/api/pick-files", methods=["POST"])
    def pick_files():
        """Launch Windows native file picker and add chosen files to queue."""
        files = show_native_open_file_dialog()
        if files:
            added = app.batch_manager.add_files(files)
            return jsonify({"status": "success", "count": len(files), "added": added})
        return jsonify({"status": "cancelled", "count": 0, "added": []})

    @app.route("/api/pick-folder", methods=["POST"])
    def pick_folder():
        """Launch Windows native folder picker."""
        data = request.get_json(silent=True) or {}
        mode = data.get("mode", "input")  # "input" to scan for videos, "output" for destination
        title = "Select Folder Containing Videos" if mode == "input" else "Select Output Folder"
        folder = show_native_folder_dialog(title=title)

        if not folder:
            return jsonify({"status": "cancelled", "folder": None})

        if mode == "input":
            # Scan folder for video extensions
            video_exts = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".flv"}
            found_files = []
            p = Path(folder)
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in video_exts:
                    found_files.append(str(f))

            added = app.batch_manager.add_files(found_files)
            return jsonify({
                "status": "success",
                "folder": folder,
                "found_count": len(found_files),
                "added": added,
            })
        else:
            # Mode is output destination
            return jsonify({"status": "success", "folder": folder})

    @app.route("/api/add-files", methods=["POST"])
    def add_files_endpoint():
        """Add list of file paths directly (from HTML5 drag-and-drop or paths)."""
        data = request.get_json() or {}
        file_paths = data.get("file_paths", [])
        added = app.batch_manager.add_files(file_paths)
        return jsonify({"status": "success", "added_count": len(added), "added": added})

    @app.route("/api/queue", methods=["GET"])
    def get_queue():
        return jsonify(app.batch_manager.get_state())

    @app.route("/api/thumbnail/<item_id>", methods=["GET"])
    def get_thumbnail(item_id):
        for it in app.batch_manager.items:
            if it.id == item_id and it.thumbnail:
                img_data = base64.b64decode(it.thumbnail)
                return Response(img_data, mimetype="image/jpeg")
        return "", 404

    @app.route("/api/remove-item", methods=["POST"])
    def remove_item():
        data = request.get_json() or {}
        item_id = data.get("item_id")
        if item_id:
            app.batch_manager.remove_item(item_id)
        return jsonify({"status": "success"})

    @app.route("/api/clear-queue", methods=["POST"])
    def clear_queue():
        data = request.get_json() or {}
        completed_only = data.get("completed_only", False)
        app.batch_manager.clear_queue(completed_only=completed_only)
        return jsonify({"status": "success"})

    @app.route("/api/retry-item", methods=["POST"])
    def retry_item():
        data = request.get_json() or {}
        item_id = data.get("item_id")
        if item_id:
            app.batch_manager.retry_item(item_id)
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "Missing item_id"}), 400

    @app.route("/api/retry-failed", methods=["POST"])
    def retry_failed():
        app.batch_manager.retry_failed()
        return jsonify({"status": "success"})

    @app.route("/api/settings", methods=["POST"])
    def update_settings():
        data = request.get_json() or {}
        app.batch_manager.update_settings(data)
        return jsonify({"status": "success", "settings": app.batch_manager.settings})

    @app.route("/api/start", methods=["POST"])
    def start_batch():
        app.batch_manager.start_batch()
        return jsonify({"status": "success"})

    @app.route("/api/pause", methods=["POST"])
    def pause_batch():
        app.batch_manager.pause_batch()
        return jsonify({"status": "success"})

    @app.route("/api/cancel", methods=["POST"])
    def cancel_batch():
        app.batch_manager.cancel_batch()
        return jsonify({"status": "success"})

    @app.route("/api/open-in-explorer", methods=["POST"])
    def open_explorer():
        data = request.get_json() or {}
        target_path = data.get("path")
        if target_path and os.path.exists(target_path):
            open_in_explorer(target_path)
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "Path does not exist"}), 404

    @app.route("/api/create-zip", methods=["POST"])
    def create_zip():
        zip_p = app.batch_manager.create_zip_from_completed()
        if zip_p:
            latest_zip_path["path"] = zip_p
            return jsonify({
                "status": "success",
                "zip_path": zip_p,
                "filename": Path(zip_p).name,
            })
        return jsonify({"status": "error", "message": "No completed videos found to zip"}), 400

    @app.route("/api/download-zip", methods=["GET"])
    def download_zip():
        if latest_zip_path["path"] and os.path.exists(latest_zip_path["path"]):
            return send_file(
                latest_zip_path["path"],
                as_attachment=True,
                download_name=Path(latest_zip_path["path"]).name,
            )
        return jsonify({"status": "error", "message": "No zip file generated yet"}), 404

    @app.route("/api/logs", methods=["GET"])
    def get_logs():
        return jsonify({"logs": app.batch_manager.log_history})

    @app.route("/api/progress-stream")
    def progress_stream():
        """Server-Sent Events endpoint streaming real-time queue & progress updates."""
        def event_generator():
            q = app.batch_manager.subscribe()
            try:
                # Send initial state
                initial = json.dumps({"type": "init", "data": app.batch_manager.get_state()})
                yield f"data: {initial}\n\n"

                while True:
                    try:
                        msg = q.get(timeout=25)
                        yield f"data: {json.dumps(msg)}\n\n"
                    except Exception:
                        # Keep-alive heartbeat comment
                        yield ": keep-alive\n\n"
            finally:
                app.batch_manager.unsubscribe(q)

        return Response(
            stream_with_context(event_generator()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return app
