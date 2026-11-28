"""Live upload demo: FFmpeg + YuNet + Light-ASD + Pyannote + Whisper/encoder."""
from collections import deque
from pathlib import Path
import subprocess, threading, uuid, sys
import cv2, numpy as np, torch
from flask import Flask, request, jsonify, send_file
from python_speech_features import mfcc

ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent
ASD=ROOT/'Light-ASD'; WEIGHT=ASD/'weight'/'pretrain_AVA_CVPR.model'; YUNET=ROOT/'face_detection_yunet_2023mar.onnx'
FPS,W,H,SR=25,640,360,16000; HOP,CONTEXT=12,37
app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=1024**3
jobs={}; lock=threading.Lock(); engine=None; emotion_runtime=None

def update(job,**kw):
    with lock: job.update(kw)

def ffmpeg(args):
    p=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*map(str,args)],capture_output=True)
    if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace')[-2000:])
    return p.stdout

def setup():
    if not WEIGHT.is_file(): raise RuntimeError(f'Missing Light-ASD checkpoint: {WEIGHT}')
    if not YUNET.is_file(): raise RuntimeError(f'Missing YuNet model: {YUNET}')
    sys.path.insert(0,str(ASD)); from model.Model import ASD_Model; from loss import lossAV
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); net,head=ASD_Model().to(dev),lossAV().to(dev)
    state=torch.load(WEIGHT,map_location='cpu',weights_only=True); state={k.removeprefix('module.'):v for k,v in state.items()}
    net.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith('model.')}); head.load_state_dict({k[7:]:v for k,v in state.items() if k.startswith('lossAV.')})
    net.eval(); head.eval(); detector=cv2.FaceDetectorYN.create(str(YUNET),'',(W,H),.65,.3,5000)
    return net,head,detector,dev

def crop(frame,b):
    x1,y1,x2,y2=b; s=max(x2-x1,y2-y1)/2; cx,cy=(x1+x2)/2,(y1+y2)/2
    x1,y1,x2,y2=round(cx-1.4*s),round(cy-s),round(cx+1.4*s),round(cy+1.8*s); sx,sy=224/max(1,x2-x1),224/max(1,y2-y1)
    m=np.array([[sx,0,-x1*sx],[0,sy,-y1*sy]],np.float32)
    return cv2.warpAffine(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),m,(224,224),borderMode=cv2.BORDER_CONSTANT,borderValue=110)[56:168,56:168].copy()

def overlap(a,b):
    x,y=max(a[0],b[0]),max(a[1],b[1]); r,d=min(a[2],b[2]),min(a[3],b[3]); inter=max(0,r-x)*max(0,d-y)
    return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

@torch.inference_mode()
def asd_scores(rows,audio):
    net,head,_,dev=engine; n=len(rows); pcm=audio[rows[0][0]*640:rows[0][0]*640+n*640]; pcm=np.pad(pcm,(0,max(0,n*640+240-len(pcm))))
    af=mfcc(pcm,SR,numcep=13,winlen=.025,winstep=.01)[:4*n]; af=np.pad(af,((0,max(0,4*n-len(af))),(0,0)))[:4*n]
    a=torch.tensor(af,dtype=torch.float32,device=dev)[None]; v=torch.tensor(np.stack([r[2] for r in rows]),dtype=torch.float32,device=dev)[None]
    out=net.forward_audio_visual_backend(net.forward_audio_frontend(a),net.forward_visual_frontend(v))
    return head.FC(out).softmax(-1)[:,1].cpu().numpy()

def infer_segment(tid,rows,audio,seg,score):
    global emotion_runtime
    if emotion_runtime is None:
        from emotion_runtime.emotion_pipeline import EmotionPipeline
        emotion_runtime=EmotionPipeline(load_whisper=True)
    from PIL import Image
    wav=audio[max(0,int(seg['start']*SR)):min(len(audio),int(seg['end']*SR))]
    crops=[Image.fromarray(r[2]) for r in rows]
    result=emotion_runtime.predict_track(tid,wav,crops,speaker=0,speaking=score>=.65,asd_score=score)
    result.update({'speaker':seg['speaker'],'start':float(seg['start']),'end':float(seg['end']),'visual_track':tid,'mean_asd':score})
    return result


def process(job):
    global engine
    cap=None
    try:
        update(job,status='preparing')
        if engine is None: engine=setup()
        playback=job['video']
        ffmpeg(['-i',job['src'],'-map','0:v:0','-map','0:a:0','-vf',f'fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1','-af','aresample=16000:async=1:first_pts=0','-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p','-c:a','aac','-ac','1','-ar','16000','-movflags','+faststart',playback])
        update(job,video_ready=True)
        audio=np.frombuffer(ffmpeg(['-i',playback,'-vn','-ac','1','-ar','16000','-f','s16le','pipe:1']),dtype='<i2').astype(np.float32)/32768
        from emotion_runtime.diarization_runtime import DiarizationRuntime
        segments=DiarizationRuntime(PROJECT).diarize(audio)['exclusive']; cap=cv2.VideoCapture(str(playback)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        update(job,status='running',total=total,segments=segments)
        tracks={}; records=[]; next_id=0; frame=0
        while True:
            ok,img=cap.read()
            if not ok: break
            _,ds=engine[2].detect(img); boxes=[]
            for d in ds if ds is not None else []:
                x,y,w,h=map(float,d[:4]); b=[max(0,round(x)),max(0,round(y)),min(W,round(x+w)),min(H,round(y+h))]
                if b[2]-b[0]>=16 and b[3]-b[1]>=16: boxes.append(b)
            matches=sorted((overlap(t['box'],b),tid,j) for tid,t in tracks.items() if t['last']==frame-1 for j,b in enumerate(boxes))[::-1]
            assigned={}; used=set(); usedb=set()
            for ov,tid,j in matches:
                if ov<.25: break
                if tid not in used and j not in usedb: assigned[j]=tid; used.add(tid); usedb.add(j)
            for j,b in enumerate(boxes):
                tid=assigned.get(j)
                if tid is None: tid=next_id; next_id+=1; tracks[tid]={'rows':deque(maxlen=CONTEXT)}
                tracks[tid].update(box=b,last=frame); tracks[tid]['rows'].append((frame,b,crop(img,b)))
            if frame%HOP==HOP-1:
                for tid,t in tracks.items():
                    rows=list(t['rows'])
                    if len(rows)>=5: records.extend((tid,row,float(p)) for row,p in zip(rows,asd_scores(rows,audio)))
            frame+=1
            if frame%25==0: update(job,processed=frame)
        events=[]
        for seg in segments:
            lo,hi=int(seg['start']*FPS),int(seg['end']*FPS); grouped={}
            for tid,row,p in records:
                if lo<=row[0]<=hi: grouped.setdefault(tid,[]).append((row,p))
            if not grouped: continue
            tid=max(grouped,key=lambda k:float(np.mean([x[1] for x in grouped[k]])))
            chosen=grouped[tid]; rows=[x[0] for x in chosen]; score=float(np.mean([x[1] for x in chosen]))
            events.append(infer_segment(tid,rows,audio,seg,score)); update(job,emotion_events=list(events),processed_segments=len(events))
        update(job,status='done',processed=frame,emotion_events=events)
    except Exception as exc: update(job,status='error',error=str(exc))
    finally:
        if cap is not None: cap.release()


@app.get('/')
def home(): return send_file(ROOT/'index.html')

@app.post('/api/upload')
def upload():
    f=request.files.get('video')
    if not f or not f.filename: return jsonify(error='Chưa chọn video'),400
    jid=uuid.uuid4().hex; folder=ROOT/'runs'/jid; folder.mkdir(parents=True); src=folder/'input.video'; f.save(src)
    job={'src':src,'video':folder/'playback.mp4','status':'queued','video_ready':False,'processed':0,'total':0,'emotion_events':[],'error':None}
    with lock: jobs[jid]=job
    threading.Thread(target=process,args=(job,),daemon=True).start(); return jsonify(id=jid)

@app.get('/api/status/<jid>')
def status(jid):
    with lock: job=jobs.get(jid)
    if not job: return jsonify(error='Not found'),404
    return jsonify(status=job['status'],video_ready=job.get('video_ready',False),processed=job.get('processed',0),total=job.get('total',0),emotion_events=job.get('emotion_events',[]),error=job.get('error'))

@app.get('/api/video/<jid>')
def video(jid):
    with lock: job=jobs.get(jid)
    if not job or not job.get('video_ready') or not job['video'].is_file(): return '',404
    return send_file(job['video'],mimetype='video/mp4',conditional=True)

if __name__=='__main__': app.run(host='127.0.0.1',port=8000,threaded=True,debug=False)
