/* DiGemo Emotion Studio: permission-based in-browser capture, actual server inference. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const names = {neutral:'Neutral',surprise:'Surprise',fear:'Fear',sadness:'Sadness',joy:'Joy',disgust:'Disgust',anger:'Anger'};
  const faces = {neutral:'😐',surprise:'😮',fear:'😨',sadness:'😔',joy:'😊',disgust:'😖',anger:'😠'};
  const state = {
    ready: false, sessionId: null, stream: null, recorder: null,
    analyzer: null, audioContext: null, animation: null,
    chunks: [], samples: [], sampler: null, timerInterval: null, startedAt: 0,
    active: false, processing: false, turnLimit: 12, history: [],
  };
  const api = async (path, options = {}) => {
    const r = await fetch(path, options);
    let data;
    try { data = await r.json(); } catch (_) { throw new Error(`HTTP ${r.status} — máy chủ không phản hồi JSON.`); }
    if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${r.status}`);
    return data;
  };
  const formatTime = (seconds) => {
    const s = Math.max(0, Math.floor(seconds));
    return `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
  };
  const toast = (msg) => { $('toastText').textContent = msg; $('toast').classList.remove('hidden'); };
  $('toastClose').addEventListener('click', () => $('toast').classList.add('hidden'));
  const setCaptureState = (text, mode = '') => {
    $('captureState').textContent = text;
    $('captureState').className = `tag ${mode}`;
    $('recBadge').classList.toggle('active', mode === 'recording');
    $('recText').textContent = mode === 'recording' ? 'REC' : mode === 'ready' ? 'LIVE' : 'READY';
  };
  function setProcessing(value) {
    state.processing = value;
    $('recordBtn').disabled = value || !state.stream || !state.ready;
    $('connectBtn').disabled = value || state.active;
    $('micSelect').disabled = value || state.active;
    $('cameraSelect').disabled = value || state.active;
    $('resetBtn').disabled = value || state.active;
    $('recordLabel').textContent = value ? 'Đang phân tích…' : state.active ? 'Kết thúc lượt' : 'Bắt đầu nói';
    if (value) setCaptureState('ĐANG XỬ LÝ');
    else if (state.stream) setCaptureState('SẴN SÀNG', 'ready');
  }
  const stopStream = async () => {
    if (state.stream) state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
    $('video').srcObject = null;
    $('cameraView').classList.remove('has-video');
    $('cameraPlaceholder').classList.remove('hidden');
    if (state.animation) cancelAnimationFrame(state.animation);
    if (state.audioContext) { await state.audioContext.close().catch(() => {}); state.audioContext = null; }
    state.analyzer = null;
    $('recordBtn').disabled = true;
  };
  const populateDevices = async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const devices = await navigator.mediaDevices.enumerateDevices();
    for (const [selectId, kind] of [['cameraSelect','videoinput'], ['micSelect','audioinput']]) {
      const el = $(selectId); const selection = el.value;
      el.replaceChildren(new Option('Mặc định', ''));
      let k = 0;
      for (const d of devices.filter(x => x.kind === kind)) {
        el.add(new Option(d.label || `${kind === 'videoinput' ? 'Camera' : 'Microphone'} ${++k}`, d.deviceId));
      }
      if ([...el.options].some(o => o.value === selection)) el.value = selection;
    }
  };
  const drawMicLevel = () => {
    const canvas = $('wave'), ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);
    const N = 25;
    let levels = new Array(N).fill(.07);
    if (state.analyzer) {
      const values = new Uint8Array(state.analyzer.frequencyBinCount);
      state.analyzer.getByteFrequencyData(values);
      levels = levels.map((_,i) => Math.min(1, .07 + values[Math.floor(i * values.length / (N * .45))] / 130));
    }
    for (let i=0;i<N;i++) {
      const x = i*(w/N)+2, bar = Math.max(3, levels[i]*h*.86), y=(h-bar)/2;
      ctx.fillStyle = state.active ? '#ff7b97' : '#66daca';
      ctx.beginPath();
      const bw = Math.max(2, w/N-4);
      ctx.roundRect(x,y,bw,bar,2);
      ctx.fill();
    }
    state.animation = requestAnimationFrame(drawMicLevel);
  };
  async function connectDevices() {
    if (!state.ready) throw new Error('Backend chưa sẵn sàng.');
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Trình duyệt yêu cầu localhost hoặc HTTPS để mở webcam/microphone.');
    if (state.active || state.processing) return;
    $('connectBtn').disabled = true;
    try {
      await stopStream();
      const cameraId = $('cameraSelect').value, micId = $('micSelect').value;
      const video = {width:{ideal:640},height:{ideal:360},frameRate:{ideal:15}};
      if (cameraId) video.deviceId = {exact:cameraId};
      const audio = {echoCancellation:false,noiseSuppression:false,autoGainControl:false};
      if (micId) audio.deviceId = {exact:micId};
      const stream = await navigator.mediaDevices.getUserMedia({video,audio});
      state.stream = stream;
      $('video').srcObject = stream;
      await $('video').play();
      $('cameraView').classList.add('has-video');
      $('cameraPlaceholder').classList.add('hidden');
      state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const audioSource = state.audioContext.createMediaStreamSource(stream);
      state.analyzer = state.audioContext.createAnalyser();
      state.analyzer.fftSize = 512;
      audioSource.connect(state.analyzer);
      drawMicLevel();
      await populateDevices();
      const micTrack = stream.getAudioTracks()[0];
      $('micStatus').textContent = micTrack?.label || 'Microphone connected';
      $('connectBtn').querySelector('span').textContent = 'Đổi thiết bị';
      $('recordBtn').disabled = false;
      setCaptureState('SẴN SÀNG','ready');
    } catch (err) {
      await stopStream();
      setCaptureState('LỖI THIẾT BỊ');
      throw new Error(`Không mở được camera/microphone: ${err.message}`);
    } finally {
      $('connectBtn').disabled = false;
    }
  }
  async function grabFrame() {
    const video = $('video');
    if (!video.videoWidth || !video.videoHeight) return;
    const can = document.createElement('canvas');
    const maxW=480, scale = Math.min(1,maxW/video.videoWidth);
    can.width = Math.max(32,Math.round(video.videoWidth*scale));
    can.height = Math.max(32,Math.round(video.videoHeight*scale));
    can.getContext('2d').drawImage(video,0,0,can.width,can.height);
    const blob = await new Promise(resolve => can.toBlob(resolve,'image/jpeg',.79));
    if (blob && state.samples.length < 90) state.samples.push(blob);
  }
  async function recordTurn() {
    if (state.active) return finishTurn();
    if (!state.stream || state.processing || !state.ready || !state.sessionId) return;
    if ($('manualMode').checked && !$('manualText').value.trim()) {
      toast('Chế độ nhập thủ công: hãy nhập câu tiếng Anh trước khi ghi.'); return;
    }
    try {
      const audioStream = new MediaStream(state.stream.getAudioTracks());
      const candidates=['audio/webm;codecs=opus','audio/ogg;codecs=opus','audio/webm','audio/mp4'];
      const mime = candidates.find(t => MediaRecorder.isTypeSupported(t));
      state.recorder = new MediaRecorder(audioStream, mime ? {mimeType:mime} : undefined);
      state.chunks = []; state.samples = [];
      state.recorder.ondataavailable = event => {if(event.data.size) state.chunks.push(event.data);};
      state.recorder.onerror = event => toast(`Lỗi MediaRecorder: ${event.error?.message || 'Unknown error'}`);
      state.active = true;
      state.startedAt = performance.now();
      state.recorder.start(200);
      await grabFrame();
      state.sampler = setInterval(grabFrame,240);
      state.timerInterval = setInterval(() => {
        const elapsed=(performance.now()-state.startedAt)/1000;
        $('timer').textContent=formatTime(elapsed);
        if(elapsed>=state.turnLimit) finishTurn().catch(e=>toast(e.message));
      },100);
      setCaptureState('ĐANG GHI ÂM','recording');
      $('recordLabel').textContent='Kết thúc lượt';
      $('recordBtn').classList.add('recording');
      $('recordBtn').querySelector('use').setAttribute('href','#i-stop');
      $('connectBtn').disabled = true;
      $('micSelect').disabled = true;
      $('cameraSelect').disabled = true;
      $('resetBtn').disabled = true;
    } catch(e) {
      state.active = false;
      throw e;
    }
  }
  async function finishTurn() {
    if (!state.active) return;
    state.active = false;
    clearInterval(state.sampler); clearInterval(state.timerInterval);
    $('recordBtn').classList.remove('recording');
    $('recordBtn').querySelector('use').setAttribute('href','#i-mic');
    const rec = state.recorder;
    if (rec?.state === 'recording') {
      await new Promise((resolve,reject) => {
        rec.addEventListener('stop',resolve,{once:true});
        rec.addEventListener('error',e=>reject(e.error || new Error('MediaRecorder failed')),{once:true});
        rec.stop();
      });
    }
    if (state.samples.length < 1) await grabFrame();
    const total=state.samples.length;
    if (!total || !state.chunks.length) {
      setProcessing(false); toast('Không thu được audio hoặc frame. Thử lại.'); return;
    }
    const frameCount=Math.min(8,total);
    const selected=[];
    for(let i=0;i<frameCount;i++) {
      const idx=Math.round(i*(total-1)/Math.max(frameCount-1,1));
      selected.push(state.samples[idx]);
    }
    const audioBlob = new Blob(state.chunks,{type:rec.mimeType || 'audio/webm'});
    const form = new FormData();
    form.append('session_id',state.sessionId);
    form.append('transcript',$('manualMode').checked ? $('manualText').value.trim() : '');
    form.append('audio',audioBlob,rec.mimeType.includes('ogg') ? 'audio.ogg' : 'audio.webm');
    selected.forEach((blob,i)=>form.append('frames',blob,`frame_${i}.jpg`));
    $('frameInfo').textContent=`${selected.length} FRAMES · ${(performance.now()-state.startedAt)/1000|0}s`;
    setProcessing(true);
    try {
      const data=await api('/api/predict',{method:'POST',body:form});
      renderResult(data);
      $('manualText').value='';
      setCaptureState('PHÂN TÍCH XONG','ready');
    } catch(e) {
      toast(e.message);
      setCaptureState('THỬ LẠI','ready');
    } finally {
      setProcessing(false);
      $('timer').textContent='00:00';
    }
  }
  function renderResult(data) {
    $('emptyResult').classList.add('hidden');
    $('resultContent').classList.remove('hidden');
    $('emotionIcon').textContent=faces[data.emotion] || '✨';
    $('emotionName').textContent=names[data.emotion] || data.emotion;
    $('confidenceText').textContent=`${data.confidence.toFixed(1)}%`;
    $('ringValue').textContent=`${data.confidence.toFixed(0)}%`;
    $('confidenceRing').style.setProperty('--percent',data.confidence);
    $('historyCount').textContent=data.history_size;
    $('transcriptOutput').textContent=data.transcript;
    $('transcriptSource').textContent=data.transcript_mode==='manual' ? 'MANUAL INPUT' : 'WHISPER ASR';
    const ordered=[...data.probabilities].sort((a,b)=>b.percent-a.percent);
    $('probabilityList').replaceChildren(...ordered.map((p,i)=>{
      const row=document.createElement('div');row.className=`prob-line ${i===0?'top':''}`;
      const name=document.createElement('span');name.className='prob-name';name.textContent=names[p.label] || p.label;
      const bg=document.createElement('div');bg.className='bar-bg';
      const fill=document.createElement('div');fill.className='bar-fill';fill.style.width=`${Math.max(0,Math.min(100,p.percent))}%`;
      bg.append(fill);
      const val=document.createElement('span');val.className='prob-val';val.textContent=`${p.percent.toFixed(1)}%`;
      row.append(name,bg,val);return row;
    }));
    $('asrLatency').textContent=`${data.latency_ms.asr} ms`;
    $('encoderLatency').textContent=`${data.latency_ms.encoders} ms`;
    $('modelLatency').textContent=`${data.latency_ms.digemo} ms`;
    state.history.unshift({time:new Date().toLocaleTimeString('vi-VN',{hour:'2-digit',minute:'2-digit'}),...data});
    $('turnCount').textContent=state.history.length;
    $('historyEmpty').classList.add('hidden');
    const hist=$('historyList');hist.replaceChildren(...state.history.map((item,i)=>{
      const row=document.createElement('div'); row.className='history-item';
      const seq=document.createElement('div');seq.className='hist-num';seq.textContent=String(state.history.length-i).padStart(2,'0');
      const content=document.createElement('div');content.className='hist-main';
      const title=document.createElement('strong');title.textContent=item.transcript;
      const detail=document.createElement('small');detail.textContent=`${item.time} · ${item.audio_seconds}s audio · ${item.frame_count} frames`;
      content.append(title,detail);
      const chip=document.createElement('div');chip.className='emotion-chip';chip.textContent=`${faces[item.emotion] || '•'} ${names[item.emotion] || item.emotion}`;
      const conf=document.createElement('div');conf.className='hist-confidence';conf.textContent=`${item.confidence.toFixed(1)}%`;
      row.append(seq,content,chip,conf);return row;
    }));
  }
  async function resetSession() {
    if (!state.sessionId || state.active || state.processing) return;
    $('resetBtn').disabled=true;
    try {
      await api('/api/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:state.sessionId})});
      state.history=[];
      $('historyList').replaceChildren();
      $('historyEmpty').classList.remove('hidden');
      $('turnCount').textContent='0';
      $('emptyResult').classList.remove('hidden');
      $('resultContent').classList.add('hidden');
      $('timer').textContent='00:00';
    } catch(e) {toast(e.message);} finally {$('resetBtn').disabled=false;}
  }
  async function initialize() {
    try {
      const status=await api('/api/status');
      const session=await api('/api/session',{method:'POST'});
      state.sessionId=session.session_id;
      state.ready=true;
      $('sideStatus').textContent='Mô hình sẵn sàng';
      $('sideDot').classList.remove('loading');
      $('sideDevice').textContent=status.device.toUpperCase();
      $('sideAsr').textContent=status.asr;
      $('captureHint').textContent=`Mỗi lượt tối đa ${state.turnLimit} giây`;
    } catch(e) {
      $('sideStatus').textContent='Backend chưa kết nối';
      $('sideDot').classList.add('loading');
      toast(`${e.message} — hãy chạy python digemo_web/app.py trong PowerShell.`);
      setTimeout(initialize,4500);
    }
  }
  $('connectBtn').addEventListener('click',()=>connectDevices().catch(e=>toast(e.message)));
  $('recordBtn').addEventListener('click',()=>recordTurn().catch(e=>{toast(e.message);setProcessing(false);}));
  $('resetBtn').addEventListener('click',()=>resetSession());
  $('cameraSelect').addEventListener('change',()=>{if(state.stream)connectDevices().catch(e=>toast(e.message));});
  $('micSelect').addEventListener('change',()=>{if(state.stream)connectDevices().catch(e=>toast(e.message));});
  $('manualMode').addEventListener('change',()=>{$('manualBox').classList.toggle('hidden',!$('manualMode').checked);});
  document.addEventListener('keydown',e=>{
    if(e.code!=='Space' || e.repeat || e.target.closest('input,textarea,button,select'))return;
    if(!state.stream||state.processing)return;
    e.preventDefault();recordTurn().catch(err=>toast(err.message));
  });
  window.addEventListener('beforeunload',()=>state.stream?.getTracks().forEach(t=>t.stop()));
  initialize();
})();
