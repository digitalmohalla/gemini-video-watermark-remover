#!/usr/bin/env python3
"""
Main Application Launcher for Gemini & Veo Video Watermark Remover.
Starts local Flask server and opens the native Windows Desktop Window (PyWebView)
or default web browser.
"""

import os
import sys
import time
import socket
import argparse
import threading
import webbrowser
from pathlib import Path

# Add current directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Ensure stdout/stderr exist when running with pythonw / pyw
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from core.batch_manager import BatchManager
from server.app import create_app


def find_free_port(start_port: int = 5858) -> int:
    """Find a free network port starting from start_port."""
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start_port


def main():
    parser = argparse.ArgumentParser(description="Gemini & Veo Video Watermark Remover")
    parser.add_argument("--browser", action="store_true", help="Launch in default web browser instead of desktop window")
    parser.add_argument("--port", type=int, default=0, help="Custom server port")
    parser.add_argument("--no-launch", action="store_true", help="Start server only without opening UI window")
    args = parser.parse_args()

    port = args.port if args.port > 0 else find_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    print("=" * 65)
    print("  GEMINI & VEO VIDEO WATERMARK REMOVER")
    print("  100% Local · Reverse Alpha Blending · Batch Processor")
    print("=" * 65)
    print(f"Server starting on {url}")

    # Initialize batch manager & web application
    batch_manager = BatchManager()
    app = create_app(batch_manager)

    # Disable Flask banner / request logging in terminal
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    # Run Flask in background daemon thread
    server_thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    server_thread.start()

    # Wait briefly for server to bind
    time.sleep(0.8)

    if args.no_launch:
        print(f"Application ready at: {url}")
        print("Press Ctrl+C to terminate.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down.")
            sys.exit(0)

    # Attempt to open native Desktop Window via PyWebView
    use_webview = not args.browser
    if use_webview:
        try:
            import webview

            print("Launching native Windows desktop application window...")
            window = webview.create_window(
                title="Gemini & Veo Video Watermark Remover",
                url=url,
                width=1280,
                height=840,
                min_size=(950, 620),
                background_color="#0a0d14",
            )
            webview.start()
            sys.exit(0)
        except Exception as e:
            print(f"PyWebView window failed ({e}), falling back to web browser...")

    # Fallback to browser
    print(f"Opening {url} in your default browser...")
    webbrowser.open(url)

    print("Press Ctrl+C to stop the application.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
