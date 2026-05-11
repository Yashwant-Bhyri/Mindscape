"""
MJPEG stream server for live annotated camera feed.
Runs as a daemon thread on port 5001.
Mesop embeds it via <iframe> pointing at /video_html.
"""
import threading
import time

from flask import Flask, Response

_flask_app = Flask(__name__)
_analyzer_ref = None
_server_thread = None
_server_started = False
_start_lock = threading.Lock()

_HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0a; display: flex; align-items: center; justify-content: center; height: 100vh; overflow: hidden; }
  img#feed { width: 100%; height: 100%; object-fit: cover; display: block; border-radius: 12px; }
</style>
</head>
<body>
  <img id="feed" src="/video_feed" alt="Somatic stream">
</body>
</html>"""


@_flask_app.route("/video_html")
def video_html():
    return _HTML_PAGE, 200, {"Content-Type": "text/html"}


@_flask_app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            if _analyzer_ref is None:
                time.sleep(0.05)
                continue
            frame_bytes = _analyzer_ref.get_annotated_frame_jpeg()
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
            time.sleep(0.04)  # ~25 fps cap for network efficiency

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def start(analyzer):
    """Start the Flask MJPEG server as a daemon thread (idempotent)."""
    global _analyzer_ref, _server_thread, _server_started

    with _start_lock:
        _analyzer_ref = analyzer
        if _server_started:
            return

        def _run():
            import logging
            log = logging.getLogger("werkzeug")
            log.setLevel(logging.ERROR)
            _flask_app.run(host="127.0.0.1", port=5001, threaded=True, use_reloader=False)

        _server_thread = threading.Thread(target=_run, daemon=True, name="MJPEGServer")
        _server_thread.start()
        _server_started = True
        time.sleep(0.5)  # allow Flask to bind
