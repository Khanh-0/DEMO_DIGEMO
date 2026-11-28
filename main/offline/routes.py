"""Offline metadata and browser-compatible video routes."""
from __future__ import annotations

from pathlib import Path
import subprocess
import uuid

from flask import Blueprint, current_app, jsonify, request, send_file

from .csv_sources import OfflineDataError, check_file
from .model_runner import predict_dialogue

bp = Blueprint("offline", __name__, url_prefix="/api/offline")


def _offline_video_dir() -> Path:
    path = Path(current_app.root_path) / "runs" / "offline"
    path.mkdir(parents=True, exist_ok=True)
    return path


@bp.post("/upload-video")
def upload_video():
    """Normalize AVI/MKV/etc. to browser-compatible MP4 without model encoders."""
    uploaded = request.files.get("video")
    if not uploaded or not uploaded.filename:
        return jsonify(error="Chưa chọn video offline"), 400
    job_id = uuid.uuid4().hex
    folder = _offline_video_dir() / job_id
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "input.video"
    playback = folder / "playback.mp4"
    uploaded.save(source)
    try:
        proc = subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0?", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-movflags", "+faststart", str(playback)
        ], capture_output=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr.decode(errors="replace")[-1000:])
    except Exception as exc:
        current_app.logger.exception("Offline video normalization failed")
        return jsonify(error=f"Không chuẩn hóa được video cho trình duyệt: {exc}"), 400
    return jsonify(video_id=job_id, filename=uploaded.filename, video_url=f"/api/offline/video/{job_id}")


@bp.get("/video/<video_id>")
def offline_video(video_id):
    path = _offline_video_dir() / video_id / "playback.mp4"
    if not path.is_file():
        return jsonify(error="Offline video không tồn tại"), 404
    return send_file(path, mimetype="video/mp4", conditional=True)


@bp.post("/check-file")
def check_file_route():
    """Validate uploaded-video metadata against the fixed test CSV sources."""
    data = request.get_json(silent=True) or {}
    try:
        result = check_file(
            dataset=str(data.get("dataset", "")),
            filename=str(data.get("filename", "")),
            dialogue_id=data.get("dialogue_id"),
        )
    except OfflineDataError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(result=result)


@bp.post("/predict")
def predict_route():
    """Run a model only on precomputed PKL/test features for one checked dialogue."""
    data = request.get_json(silent=True) or {}
    try:
        results = predict_dialogue(
            dataset=str(data.get("dataset", "")),
            model=str(data.get("model", "")),
            dialogue_id=str(data.get("dialogue_id", "")),
        )
    except (ValueError, FileNotFoundError) as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        # Never return Flask's HTML 500 page to the JSON frontend.
        current_app.logger.exception("Offline prediction failed")
        return jsonify(error=f"Offline prediction failed: {exc}"), 500
    return jsonify(results=results, count=len(results))
