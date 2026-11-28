"""Orchestrates per-track Whisper, IEMOCAP encoders and CRG-3."""
from .config import RuntimeConfig
from .crg3_runtime import CRG3Runtime
from .diarization_runtime import DiarizationRuntime
from .iemocap_encoders import IEMOCAPEncoders
from .track_buffer import TrackBufferStore


class EmotionPipeline:
    def __init__(self, config=None, load_whisper=False):
        self.config = config or RuntimeConfig()
        if not self.config.is_iemocap:
            raise NotImplementedError("MELD mode needs a DiGemo runtime adapter; IEMOCAP mode is CRG-3")
        self.encoders = IEMOCAPEncoders(self.config)
        self.crg3 = CRG3Runtime(self.config)
        self.buffers = TrackBufferStore(self.config.fps, self.config.max_track_seconds)
        self.diarizer = DiarizationRuntime(self.config.project_root)
        self.whisper = None
        if load_whisper:
            from .whisper_runtime import WhisperRuntime
            self.whisper = WhisperRuntime(self.config.whisper_model, self.config.resolved_device())

    def predict_track(self, track_id, audio, face_crops, transcript="", speaker=0,
                      speaking=True, asd_score=0.0):
        track = self.buffers.update_status(track_id, speaking, asd_score, 0)
        for crop in face_crops:
            track.append_crop(crop, 0)
        track.append_audio(audio)
        if not transcript and self.whisper is not None:
            transcript = self.whisper.transcribe(track.audio_array())
        if transcript:
            track.text_history.append(transcript)
        text_feature = self.encoders.text(transcript)
        audio_feature = self.encoders.audio(track.audio_array())
        visual_feature = self.encoders.visual(list(track.crops))
        result = self.crg3.predict(text_feature, audio_feature, visual_feature, speaker)
        track.emotion = result["emotion"]
        track.confidence = result["confidence"]
        result.update({"track_id": int(track_id), "text": transcript,
                       "speaking": bool(speaking), "asd_score": float(asd_score)})
        return result
