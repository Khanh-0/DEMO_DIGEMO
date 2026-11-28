"""Playback-gated Online Streaming pilot.

This namespace never uses a local browser Object URL. The browser receives
only backend-normalized MP4 and a processing watermark.
"""
from __future__ import annotations
from pathlib import Path
import json
import traceback
import subprocess
import threading
import time
import uuid
import shutil
import wave
import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request, send_file
from .asd_stream import ASDStream
from .emotion_stream import StreamingEmotion, MELDStreamingEmotion
from .inference_worker import InferenceWorker
from .diart_worker import DiartWorker

bp = Blueprint("online_streaming", __name__, url_prefix="/api/online-stream")
_jobs = {}
_lock = threading.Lock()
FPS = 30
# Streaming block: 12 frames = 0.4 seconds at 30 FPS.
BLOCK_FRAMES = 12

def _root():
    p = Path(current_app.root_path) / "runs" / "online_streaming"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cleanup_old_jobs(keep=5):
    jobs = sorted(_root().iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
    protected = {Path(job["folder"]).resolve() for job in _jobs.values() if job.get("status") in {"queued", "normalizing", "processing"}}
    for folder in jobs[keep:]:
        if folder.is_dir() and folder.resolve() not in protected:
            shutil.rmtree(folder, ignore_errors=True)

def _update(jid, **values):
    with _lock:
        job = _jobs[jid]
        job.update(values)
        folder = job.get("folder")
        if folder:
            try:
                (Path(folder) / "status.json").write_text(json.dumps(_json_safe(job), ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass


def _trace(jid, **values):
    job = _jobs.get(jid)
    if not job or not job.get("folder"): return
    record = {"wall_time": time.time(), **values}
    try:
        with (Path(job["folder"]) / "debug_trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")
    except Exception:
        pass


def _json_safe(value):
    """Convert runtime-only values to JSON-safe status data."""
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)): return [_json_safe(item) for item in value]
    return value


def _write_snapshot(folder, snapshot):
    """Persist debug artifacts so snapshot/track alignment can be inspected."""
    folder.mkdir(parents=True, exist_ok=True)
    sid = int(snapshot["snapshot_id"])
    prefix = folder / f"snapshot_{sid:04d}"
    wav_path = prefix.with_suffix(".wav")
    pcm = np.asarray(snapshot["audio"], dtype=np.float32)
    pcm16 = np.clip(pcm, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    with wave.open(str(wav_path), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(pcm16.tobytes())
    feature = snapshot.get("face_feature")
    feature_file = None
    if feature is not None:
        feature_file = prefix.with_suffix(".face_feature.npy").name
        np.save(folder / feature_file, np.asarray(feature, dtype=np.float32))
    crop_paths = []
    for index, crop in enumerate(snapshot.get("crops", [])):
        path = folder / f"snapshot_{sid:04d}_crop_{index:02d}.jpg"
        cv2.imwrite(str(path), np.asarray(crop))
        crop_paths.append(path.name)
    metadata = {key: value for key, value in snapshot.items() if key not in ("audio", "crops", "face_feature")}
    metadata.update({"audio_file": wav_path.name, "crop_files": crop_paths, "face_feature_file": feature_file, "audio_samples": int(len(pcm))})
    prefix.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

def _work(jid):
    job = _jobs[jid]
    try:
        _update(jid, status="normalizing", worker_alive=True, last_update_at=time.time())
        proc = subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(job["source"]),
            "-map", "0:v:0", "-map", "0:a:0?", "-vf", "fps=30,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-movflags", "+faststart", str(job["playback"])
        ], capture_output=True)
        if proc.returncode: raise RuntimeError(proc.stderr.decode(errors="replace")[-1200:])
        cap = cv2.VideoCapture(str(job["playback"]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        audio_wav = Path(job["folder"]) / "audio_16k.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(job["playback"]), "-vn", "-ac", "1", "-ar", "16000", str(audio_wav)], check=True)
        audio_bytes = audio_wav.read_bytes()[44:]
        audio = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
        dataset = job.get("dataset", "IEMOCAP")
        asd = ASDStream(job["project_root"])
        _update(jid, status="processing", video_ready=True, total_frames=total, total_duration=total/fps if total else 0, processed_frame=0, processed_until=0.0, worker_alive=True, last_update_at=time.time(), speaker_source="Light-ASD audio-visual active speaker")
        frame = 0; snapshot_seq = 0; diart_seen = set(); diart_segments = []; diart_processed = set()

        def on_diart_segment(segment):
            key = (segment["speaker"], round(segment["start"], 2), round(segment["end"], 2))
            if key not in diart_seen:
                diart_seen.add(key); diart_segments.append(segment)
                _update(jid, diart_segments=len(diart_segments), last_diart_segment=segment, diart_completed=_jobs[jid].get("diart_completed", 0) + 1)

        def on_diart_error(error):
            _update(jid, diart_error=error, diart_failed=_jobs[jid].get("diart_failed", 0) + 1)

        diart = DiartWorker(audio_wav, on_diart_segment, on_diart_error)
        _update(jid, diart_backend="Diart", diart_submitted=1, diart_completed=0, diart_failed=0, diart_segments=0)
        processing_started = time.perf_counter()
        snapshot_dir = Path(job["folder"]) / "snapshots"
        _update(jid, frames_seen=0, faces_detected=0, tracks_created=0, asd_submitted=0, asd_completed=0, asd_failed=0, asd_dropped=0, asd_max_score=0.0, utterances_started=0, utterances_closed=0, snapshots_dir=str(snapshot_dir))

        def make_runtime():
            return MELDStreamingEmotion(job["project_root"]) if str(dataset).upper() == "MELD" else StreamingEmotion("IEMOCAP")

        def run_snapshot(runtime, snapshot):
            result = runtime.predict(snapshot["audio"], snapshot["crops"], snapshot["track_id"], snapshot["start"], snapshot["end"], snapshot["asd_score"], snapshot.get("diart_speaker"))
            result["snapshot_id"] = snapshot.get("snapshot_id")
            return result

        def on_inference_result(result):
            with _lock:
                if result.get("error"):
                    _jobs[jid]["inference_error"] = result["error"]
                    _jobs[jid]["inference_failed"] = _jobs[jid].get("inference_failed", 0) + 1
                else:
                    _jobs[jid].setdefault("emotion_events", []).append(result)
                    _jobs[jid]["inference_completed"] = _jobs[jid].get("inference_completed", 0) + 1
                    _jobs[jid]["whisper_text_length"] = len(result.get("text", ""))
                    _jobs[jid]["whisper_device"] = result.get("whisper_device", "unknown")
                    _jobs[jid]["last_inference_snapshot"] = result.get("snapshot_id")
                    events_path = Path(_jobs[jid]["folder"]) / "events.jsonl"
                    with events_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                _jobs[jid]["last_update_at"] = time.time()

        inference = InferenceWorker(make_runtime, run_snapshot, on_inference_result, maxsize=4)
        _update(jid, asd_backend="Light-ASD", inference_queue_size=0, inference_completed=0, inference_failed=0, inference_dropped=0, snapshots_created=0, snapshots_enqueued=0, utterances_started=0)
        _update(jid, asd_backend="Light-ASD", max_gap_frames=asd.max_gap_frames)

        def close_utterance(state, end_frame):
            nonlocal snapshot_seq
            if state is None or end_frame / fps - state["start"] < .5: return
            snapshot_seq += 1
            track = asd.tracks.get(int(state["track_id"]))
            crops = list(asd.crops_for_track(state["track_id"], end_frame, seconds=1.0, count=8))
            snapshot = {"snapshot_id": snapshot_seq, "track_id": state["track_id"], "speaker_id": state.get("speaker_id", f"SPEAKER_{int(state['track_id']):02d}"),
                "diart_speaker": state.get("diart_speaker"), "start": state["start"], "end": end_frame/fps, "asd_score": state["asd_sum"] / max(1, state["blocks"]), "box_history": list(getattr(track, "box_history", [])), "tracking_box": list(getattr(track, "box", [])), "display_box": list(getattr(track, "display_box", getattr(track, "box", []))), "face_feature": getattr(track, "embedding_centroid", None), "audio": audio[max(0, int(state["start"]*16000)):min(len(audio), int(end_frame/fps*16000))].copy(), "crops": crops, "debug_folder": str(job["folder"])}
            _write_snapshot(snapshot_dir, snapshot)
            _update(jid, snapshots_created=snapshot_seq, last_snapshot=snapshot["snapshot_id"], last_snapshot_crops=len(snapshot["crops"]))
            if not inference.submit(snapshot):
                _update(jid, inference_dropped=_jobs[jid].get("inference_dropped", 0) + 1, inference_warning="Inference queue đầy; tracking vẫn tiếp tục")
            else:
                _update(jid, snapshots_enqueued=_jobs[jid].get("snapshots_enqueued", 0) + 1)
            _update(jid, inference_queue_size=inference.queue.qsize(), last_update_at=time.time())

        while frame < total:
            block_started = time.perf_counter()
            end_frame = min(total, frame + BLOCK_FRAMES)
            scored_tracks = asd.process_block(cap, audio, frame, end_frame)
            asd_elapsed = time.perf_counter() - block_started
            for track in scored_tracks:
                track["asd_score"] = float(track.get("asd_score", 0.0))
            asd_elapsed = time.perf_counter() - block_started
            active = max(scored_tracks, key=lambda x: x["asd_score"]) if scored_tracks else None
            latest_tracks = [{**track, "speaker": f"SPEAKER_{track['track_id']:02d}"} for track in scored_tracks]
            # Fuse finalized Diart audio turns with Light-ASD evidence only.
            for index, segment in enumerate(list(diart_segments)):
                if index in diart_processed or segment["end"] > end_frame / fps: continue
                candidates = [t for t in scored_tracks if t["asd_score"] >= 0.25]
                chosen = max(candidates or scored_tracks, key=lambda x: x.get("asd_score", 0.0)) if (candidates or scored_tracks) else None
                if chosen is not None and segment["end"] - segment["start"] > 0.0:
                    close_utterance({"track_id": int(chosen["track_id"]), "speaker_id": segment["speaker"], "diart_speaker": segment["speaker"], "start": max(0.0, segment["start"] - .2), "end": segment["end"] + .2, "asd_sum": float(chosen.get("asd_score", 0.0)), "blocks": 1}, int(segment["end"] * fps))
                    _update(jid, utterances_closed=_jobs[jid].get("utterances_closed", 0) + 1, last_diart_fused=segment)
                diart_processed.add(index)
            if active:
                _update(jid, asd_max_score=max(_jobs[jid].get("asd_max_score", 0.0), float(active["asd_score"])))
            frame = end_frame
            _update(jid, frames_seen=frame, faces_detected=sum(len(track.rows) for track in asd.tracks.values()), tracks_created=asd.next_id, processed_frame=frame, processed_until=frame/fps, worker_alive=True, last_update_at=time.time(), latest_tracks=latest_tracks, current_speaker=(f"SPEAKER_{active['track_id']:02d}" if active else None), cache_tracks=len(asd.tracks), cache_reid=len(asd.reid_cache), message="Diart audio turns + Light-ASD fusion", last_asd_seconds=asd_elapsed)
            _trace(jid, frame=frame, video_time=frame/fps, track_count=len(scored_tracks), tracks=[{"track_id": int(t["track_id"]), "rows": len(asd.tracks[int(t["track_id"])] ["rows"]), "score": float(t["asd_score"]), "speaking": bool(t["speaking"])} for t in scored_tracks], utterances_started=_jobs[jid].get("utterances_started", 0), snapshots_created=_jobs[jid].get("snapshots_created", 0), inference_completed=_jobs[jid].get("inference_completed", 0), diart_segments=len(diart_segments), diart_processed=len(diart_processed))
            # Keep the Online route at wall-clock video speed while model workers run independently.
            sleep_for = frame / fps - (time.perf_counter() - processing_started)
            if sleep_for > 0: time.sleep(sleep_for)
        diart.join(timeout=30)
        for index, segment in enumerate(list(diart_segments)):
            if index not in diart_processed and segment["end"] - segment["start"] > 0.0:
                candidates = list(asd.tracks.values())
                if candidates:
                    chosen = max(candidates, key=lambda t: float(getattr(t, "asd_score", 0.0)))
                    close_utterance({"track_id": chosen.track_id, "speaker_id": segment["speaker"], "diart_speaker": segment["speaker"], "start": max(0.0, segment["start"] - .2), "asd_sum": float(getattr(chosen, "asd_score", 0.0)), "blocks": 1}, int(segment["end"] * fps))
        inference.queue.join()
        inference.close()
        cap.release()
        with _lock: final_events = list(_jobs[jid].get("emotion_events", []))
        _update(jid, status="done", processed_frame=total, processed_until=total/fps if total else 0.0, worker_alive=False, last_update_at=time.time(), emotion_events=final_events, inference_queue_size=0)
    except Exception as exc:
        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            (Path(job["folder"]) / "traceback.txt").write_text(error_text, encoding="utf-8")
        except Exception:
            pass
        _update(jid, status="error", error=str(exc), traceback_file="traceback.txt", worker_alive=False, last_update_at=time.time(), stage=job.get("stage", "unknown"))

@bp.post("/upload")
def upload():
    uploaded = request.files.get("video")
    dataset = str(request.form.get("dataset", "IEMOCAP")).upper()
    if dataset not in ("IEMOCAP", "MELD"): return jsonify(error="Dataset Online phải là IEMOCAP hoặc MELD"), 400
    if not uploaded or not uploaded.filename: return jsonify(error="Chưa chọn video"), 400
    _cleanup_old_jobs()
    jid = uuid.uuid4().hex; folder = _root()/jid; folder.mkdir(parents=True)
    source, playback = folder/"input.video", folder/"playback.mp4"; uploaded.save(source)
    with _lock:
        _jobs[jid] = {"folder": folder, "source": source, "playback": playback, "project_root": str(Path(current_app.root_path).parent), "dataset": dataset, "status": "queued", "video_ready": False, "worker_alive": False, "processed_frame": 0, "processed_until": 0.0, "last_update_at": time.time(), "latest_tracks": [], "emotion_events": [], "error": None}
    threading.Thread(target=_work, args=(jid,), daemon=True).start()
    return jsonify(job_id=jid, video_url=f"/api/online-stream/video/{jid}", status_url=f"/api/online-stream/status/{jid}")

@bp.get("/status/<jid>")
def status(jid):
    job = _jobs.get(jid)
    if not job: return jsonify(error="Not found"), 404
    public = {key: value for key, value in job.items() if key not in ("folder", "source", "playback")}
    return jsonify(_json_safe(public))

@bp.delete("/job/<jid>")
def delete_job(jid):
    job = _jobs.get(jid)
    if not job: return jsonify(error="Not found"), 404
    if job.get("status") in {"queued", "normalizing", "processing"}:
        return jsonify(error="Job đang chạy, chưa thể xóa"), 409
    folder = Path(job["folder"])
    with _lock: _jobs.pop(jid, None)
    shutil.rmtree(folder, ignore_errors=True)
    return jsonify(deleted=True, job_id=jid)


@bp.get("/video/<jid>")
def video(jid):
    job = _jobs.get(jid)
    if not job or not job.get("video_ready") or not job["playback"].is_file(): return jsonify(error="Not found"),404
    return send_file(job["playback"], mimetype="video/mp4", conditional=True)
