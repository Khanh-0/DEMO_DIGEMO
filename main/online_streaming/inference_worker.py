"""Bounded asynchronous emotion inference worker."""
from __future__ import annotations
import json
import queue
import threading
import traceback
from pathlib import Path


class InferenceWorker:
    def __init__(self, make_runtime, predict_snapshot, on_result, maxsize=4):
        self.queue = queue.Queue(maxsize=maxsize)
        self.make_runtime = make_runtime
        self.predict_snapshot = predict_snapshot
        self.on_result = on_result
        self.runtime = None
        self.stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, snapshot) -> bool:
        try:
            self.queue.put_nowait(snapshot)
            return True
        except queue.Full:
            return False

    def _run(self):
        while not self.stop:
            item = self.queue.get()
            if item is None:
                self.queue.task_done(); break
            try:
                if self.runtime is None:
                    self.runtime = self.make_runtime()
                result = self.predict_snapshot(self.runtime, item)
                self.on_result(result)
            except Exception as exc:
                detail = {"error": str(exc), "snapshot_id": item.get("snapshot_id"), "audio_samples": len(item.get("audio", [])), "crop_count": len(item.get("crops", [])), "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}
                folder = item.get("debug_folder")
                if folder:
                    try:
                        with (Path(folder) / "inference_errors.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(detail, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
                self.on_result(detail)
            finally:
                self.queue.task_done()

    def close(self):
        self.stop = True
        self.queue.put(None)
        self.thread.join(timeout=10)
