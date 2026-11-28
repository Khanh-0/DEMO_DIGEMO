"""Dataset-aware streaming emotion runners for IEMOCAP and MELD."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from PIL import Image
import torch


class StreamingEmotion:
    """IEMOCAP runtime preserved for the original 1024/1582/342 pipeline."""
    def __init__(self, dataset="IEMOCAP"):
        from emotion_runtime.emotion_pipeline import EmotionPipeline
        if str(dataset).upper() != "IEMOCAP":
            raise ValueError("StreamingEmotion is IEMOCAP-only; use MELDStreamingEmotion for MELD")
        self.runtime = EmotionPipeline(load_whisper=True)
        self.dataset = "IEMOCAP"
        self.speaker_indices = {}

    def speaker_index(self, track_id: int) -> int:
        if track_id not in self.speaker_indices: self.speaker_indices[track_id] = len(self.speaker_indices)
        return self.speaker_indices[track_id]

    def predict(self, audio, crops, track_id, start, end, asd_score, diart_speaker=None):
        wav=np.asarray(audio,np.float32).reshape(-1)
        transcript=self.runtime.whisper.transcribe(wav) if self.runtime.whisper else ""
        text_feature=self.runtime.encoders.text(transcript); audio_feature=self.runtime.encoders.audio(wav)
        visual_feature=self.runtime.encoders.visual([Image.fromarray(c) for c in crops]); speaker=self.speaker_index(int(track_id))
        result=self.runtime.crg3.predict(text_feature,audio_feature,visual_feature,speaker)
        return {"speaker":diart_speaker or f"SPEAKER_{int(track_id):02d}","diart_speaker":diart_speaker,"speaker_index":speaker,"visual_track":int(track_id),"start":float(start),"end":float(end),"text":transcript,"emotion":result["emotion"],"confidence":float(result["confidence"]),"probabilities":result.get("probabilities",[]),"mean_asd":float(asd_score),"visual_status":"face-crops" if crops else "no-face-zero-vector","model":"CRG-3","dataset":self.dataset}


class MELDStreamingEmotion:
    """MELD online runtime: Text 1024, Wav2Vec2+head Audio 300, Face 342."""
    def __init__(self, project_root):
        root=Path(project_root); runtime_root=Path(__file__).resolve().parents[1]; code=runtime_root/'encoders'/'code'; feat=runtime_root/'encoders'/'checkpoints'/'meld'; shared=runtime_root/'encoders'/'checkpoints'/'shared'
        root_path=str(root.resolve())
        if root_path not in sys.path: sys.path.insert(0,root_path)
        code_path=str(code.resolve())
        if code_path not in sys.path: sys.path.insert(0,code_path)
        from encoders_online import OnlineEncodersV2
        from faster_whisper import WhisperModel
        from CRG3.config import model_config_from_dict
        from CRG3.model import build_model
        device='cuda' if torch.cuda.is_available() else 'cpu'
        args=SimpleNamespace(encoder_dir=str(code),feat_dir=str(feat),text_ckpt=str(feat/'text_ft.pt'),face_ckpt=str(shared/'face_fer.pt'),audio_model='facebook/wav2vec2-base',audio_mode='meld')
        self.encoders=OnlineEncodersV2(args,device)
        self.whisper_device = 'cuda' if device == 'cuda' else 'cpu'
        try:
            self.whisper=WhisperModel('large-v3',device=self.whisper_device,compute_type='float16' if self.whisper_device=='cuda' else 'int8')
        except Exception:
            self.whisper_device = 'cpu'
            self.whisper=WhisperModel('large-v3',device='cpu',compute_type='int8')
        ckpt=torch.load(root/'CRG3'/'checkpoints'/'meld.pt',map_location='cpu',weights_only=False); cfg=model_config_from_dict(ckpt['model_config']); self.crg3=build_model(cfg,ckpt['feature_dims'],ckpt['num_classes']).to(device); self.crg3.load_state_dict(ckpt['model']); self.crg3.eval(); self.device=device; self.history=[]; self.speaker_indices={}; self.dataset='MELD'

    def speaker_index(self,track_id):
        if track_id not in self.speaker_indices:self.speaker_indices[track_id]=len(self.speaker_indices)
        return self.speaker_indices[track_id]

    @torch.inference_mode()
    def predict(self,audio,crops,track_id,start,end,asd_score,diart_speaker=None):
        wav=np.asarray(audio,np.float32).reshape(-1)
        try:
            segments,_=self.whisper.transcribe(wav,language='en',beam_size=3,condition_on_previous_text=False)
            text=' '.join(x.text.strip() for x in segments).strip()
        except Exception as exc:
            if self.whisper_device != 'cuda': raise
            self.whisper_device='cpu'; self.whisper=WhisperModel('large-v3',device='cpu',compute_type='int8')
            segments,_=self.whisper.transcribe(wav,language='en',beam_size=3,condition_on_previous_text=False)
            text=' '.join(x.text.strip() for x in segments).strip()
        t=self.encoders.text(text); a_feat=self.encoders.audio(wav); v=self.encoders.visual([Image.fromarray(c) for c in crops]); sp=self.speaker_index(int(track_id)); self.history.append((t,a_feat,v,sp)); n=len(self.history); T,A,V,S=zip(*self.history); batch={'text':torch.tensor(np.stack(T),dtype=torch.float32)[None].to(self.device),'audio':torch.tensor(np.stack(A),dtype=torch.float32)[None].to(self.device),'visual':torch.tensor(np.stack(V),dtype=torch.float32)[None].to(self.device),'mask':torch.ones(1,n,dtype=torch.bool,device=self.device),'speaker':torch.tensor([list(S)],dtype=torch.long,device=self.device)}
        diff,same=[],[]
        for i in range(n):
            past=list(range(i)); diff.append([j for j in past if S[j]!=S[i]]); same.append([j for j in past if S[j]==S[i]])
        def pack(rows):
            w=max(1,max((len(x) for x in rows),default=0)); idx=torch.full((1,n,w),-1,dtype=torch.long,device=self.device)
            for i,row in enumerate(rows):
                if row: idx[0,i,:len(row)]=torch.tensor(row,device=self.device)
            return {'idx':idx,'mask':idx.ge(0)}
        batch['diff']=pack(diff); batch['same']=pack(same); probs=self.crg3(batch)[0,-1].softmax(-1).cpu().numpy(); eid=int(probs.argmax())
        labels=['neutral','surprise','fear','sadness','joy','disgust','anger']; return {'speaker':diart_speaker or f'SPEAKER_{int(track_id):02d}','diart_speaker':diart_speaker,'speaker_index':sp,'visual_track':int(track_id),'start':float(start),'end':float(end),'text':text,'whisper_device':self.whisper_device,'audio_samples':int(len(wav)),'emotion':labels[eid],'confidence':float(probs[eid]),'probabilities':probs.tolist(),'mean_asd':float(asd_score),'visual_status':'face-crops' if crops else 'no-face-zero-vector','model':'CRG-3','dataset':'MELD'}
