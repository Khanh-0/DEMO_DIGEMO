(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let checked = null;
  let latestPredictions = [];
  let offlineVideoObjectUrl = null;
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const setStatus = (text, error = false) => { $('offlineStatus').textContent = text; $('offlineStatus').classList.toggle('danger', error); const pill = $('status')?.closest('.status-pill'); if (pill) { pill.classList.toggle('status-error', error); pill.classList.toggle('status-success', !error && /Hoàn tất|Đã kiểm tra/.test(text)); pill.classList.toggle('status-busy', !error && !/Hoàn tất|Đã kiểm tra|Sẵn sàng/.test(text)); } $('status').textContent = error ? 'Có lỗi' : text; };

  document.querySelectorAll('.mode-tab').forEach(tab => tab.addEventListener('click', () => {
    document.querySelectorAll('.mode-tab').forEach(x => {
      const active = x === tab;
      x.classList.toggle('active', active);
      if (active) x.setAttribute('aria-current', 'page'); else x.removeAttribute('aria-current');
    });
    document.querySelectorAll('.mode-panel').forEach(x => x.classList.toggle('active', x.id === tab.dataset.mode));
    const onlineIntro = $('onlineIntro');
    if (onlineIntro) onlineIntro.classList.toggle('hidden', tab.dataset.mode !== 'online');
  }));

  let onlineJob = null;
  let onlineJobToken = 0;
  let latestOnlineEvents = [];
  let onlineBackendDone = false;
  let onlinePlaybackEnded = false;
  function resetOnlineState() {
    onlineJobToken += 1;
    onlineJob = null;
    onlineBackendDone = false;
    onlinePlaybackEnded = false;
    latestOnlineEvents = [];
    const player = $('onlineVideo');
    player.pause();
    player.removeAttribute('src');
    delete player.dataset.streamingJob;
    player.load();
    renderOnlineEvents([]);
    $('onlineDownload').disabled = true;
  }
  function setPlayerVideoState(player) {
    const stage = player?.closest('.figma-player');
    if (stage) stage.classList.toggle('has-video', Boolean(player?.currentSrc || player?.src));
  }
  function bindPlayerState(player) {
    if (!player) return;
    ['loadedmetadata', 'canplay', 'emptied', 'error'].forEach(eventName => player.addEventListener(eventName, () => setPlayerVideoState(player)));
    setPlayerVideoState(player);
  }
  bindPlayerState($('onlineVideo'));
  bindPlayerState($('offlineVideo'));

  $('onlineFile').addEventListener('change', () => {
    const file = $('onlineFile').files[0]; const drop = $('onlineDrop');
    const title = drop?.querySelector('strong'); const label = drop?.querySelector('label');
    if (drop) drop.classList.toggle('has-file', Boolean(file));
    if (title) title.textContent = file ? file.name : 'Video nguồn';
    if (label) label.textContent = file ? '↻  Đổi video' : '↑  Chọn video';
    $('onlineStart').disabled = !file;
  });
  $('onlineStart').disabled = true;

  $('onlineStart').addEventListener('click', async () => {
    const file = $('onlineFile').files[0];
    if (!file) return setStatus('Hãy chọn video online trước.', true);
    const button = $('onlineStart');
    const player = $('onlineVideo');
    resetOnlineState();
    const requestToken = onlineJobToken;
    button.disabled = true;
    setStatus(`Đang upload và chuẩn bị playback backend…`);
    const form = new FormData();
    form.append('video', file, file.name);
    form.append('dataset', 'MELD');
    try {
      const response = await fetch('/api/online-stream/upload', { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw Error(data.error || 'Upload online thất bại');
      onlineJob = data.job_id;
      pollOnlineStream(requestToken).catch(error => { setStatus(error.message, true); buttonOnlineDone(); });
    } catch (error) {
      setStatus(error.message, true);
      button.disabled = false;
    }
  });

  async function pollOnlineStream(requestToken = onlineJobToken) {
    if (!onlineJob || requestToken !== onlineJobToken) return;
    const response = await fetch(`/api/online-stream/status/${onlineJob}`);
    const data = await response.json();
    if (!response.ok) throw Error(data.error || 'Không đọc được trạng thái online streaming');
    if (data.status === 'error') { const detail = data.error || 'Online streaming thất bại'; const stage = data.stage ? ` [stage=${data.stage}]` : ''; $('onlineStatus').textContent = `${detail}${stage}`; throw Error(`${detail}${stage}`); }
    const player = $('onlineVideo');
    if (data.video_ready && player.dataset.streamingJob !== onlineJob) {
      player.dataset.streamingJob = onlineJob;
      player.src = `/api/online-stream/video/${onlineJob}`;
      player.load();
      player.play().catch(() => setStatus('Playback đã sẵn sàng; bấm Play nếu trình duyệt chặn autoplay.'));
    }
    latestOnlineEvents = data.emotion_events || [];
    renderOnlineEvents(latestOnlineEvents);
    $('onlineDownload').disabled = latestOnlineEvents.length === 0;
    const diag = data.inference_error ? ` · inference lỗi: ${data.inference_error}` : ` · faces ${data.faces_detected || 0} · tracks ${data.tracks_created || 0} · ASD ${(Number(data.asd_max_score || 0)).toFixed(2)} · snapshot ${data.snapshots_created || 0}/${data.snapshots_enqueued || 0} · infer ${data.inference_completed || 0}`;
    const phase = data.status === 'normalizing' ? 'Đang chuẩn hóa video…' : data.status === 'processing' ? 'Đang tải model và phân tích…' : data.status === 'done' ? 'Hoàn tất phân tích' : `Online: ${data.status}`;
    setStatus(`${phase}${diag}`);
    if (data.status === 'done') onlineBackendDone = true;
    if (data.status !== 'done' && data.status !== 'error') window.setTimeout(() => pollOnlineStream(requestToken).catch(error => { setStatus(error.message, true); buttonOnlineDone(); }), 500);
    else buttonOnlineDone();
  }

  $('onlineVideo').addEventListener('ended', () => {
    onlinePlaybackEnded = true;
    setStatus(onlineBackendDone ? 'Playback finished · backend done' : 'Playback finished · backend still processing');
  });

  $('onlineDownload').addEventListener('click', () => {
    const rows = [['speaker','start','end','text','emotion','confidence','visual_track','mean_asd'], ...latestOnlineEvents.map(x => [x.speaker,x.start,x.end,x.text,x.emotion,x.confidence,x.visual_track,x.mean_asd])];
    const csv = rows.map(row => row.map(v => `"${String(v ?? '').replaceAll('"','""')}"`).join(',')).join('\\n');
    const blobUrl = URL.createObjectURL(new Blob([csv], {type:'text/csv'}));
    const link = document.createElement('a'); link.href = blobUrl; link.download = 'online_stream_events.csv'; link.click();
    window.setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
  });

  function buttonOnlineDone() { $('onlineStart').disabled = !$('onlineFile').files[0]; }
  function renderOnlineEvents(items) {
    $('onlineRows').innerHTML = items.map((event, index) => `<tr><td>${esc(event.speaker || '')}</td><td>${Number(event.start || 0).toFixed(2)}s</td><td>${Number(event.end || 0).toFixed(2)}s</td><td>${esc(event.text || '')}</td><td>${esc(event.emotion || '')} ${(Number(event.confidence || 0) * 100).toFixed(1)}%</td><td>Realtime</td></tr>`).join('');
  }

  $('offlineFile').addEventListener('change', () => {
    const file = $('offlineFile').files[0];
    if (offlineVideoObjectUrl) { URL.revokeObjectURL(offlineVideoObjectUrl); offlineVideoObjectUrl = null; }
    $('offlineVideo').removeAttribute('src'); $('offlineVideo').load();
    const drop = $('offlineDrop'); const name = $('offlineFileName'); if (drop) drop.classList.toggle('has-file', Boolean(file)); if (name) name.textContent = file ? file.name : 'Chưa có tệp';
    invalidateOfflineCheck(); setStatus(file ? `Đã chọn ${file.name}. Bấm Kiểm tra tệp để chuẩn hóa video.` : 'Chưa chọn video.');
  });

  let offlineRunToken = 0;
  let offlineModelResults = {};
  const offlineComparisons = { MELD: {}, IEMOCAP: {} };
  function invalidateOfflineCheck() {
    checked = null;
    offlineRunToken += 1;
    latestPredictions = [];
    offlineModelResults = {};
    $('offlinePredict').disabled = true;
    $('downloadCsv').disabled = true;
  }
  ['dataset', 'model', 'dialogue'].forEach(id => $(id).addEventListener('change', invalidateOfflineCheck));
  $('checkFile').addEventListener('click', async () => {
    const file = $('offlineFile').files[0];
    if (!file) return setStatus('Hãy chọn video test.', true);
    const checkToken = ++offlineRunToken;
    $('checkFile').disabled = true;
    try {
      const dataset = $('dataset').value;
      const dialogueId = $('dialogue').value.trim() || null;
      setStatus('Đang chuẩn hóa video và đối chiếu CSV/timestamp…');
      const uploadForm = new FormData(); uploadForm.append('video', file, file.name);
      const uploadResponse = await fetch('/api/offline/upload-video', {method:'POST', body:uploadForm});
      const uploadData = await uploadResponse.json();
      if (!uploadResponse.ok) throw Error(uploadData.error || 'Không chuẩn hóa được video');
      $('offlineVideo').src = uploadData.video_url; $('offlineVideo').load(); $('offlineVideo').closest('.figma-player')?.classList.add('has-video');
      const response = await fetch('/api/offline/check-file', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({dataset, filename:file.name, dialogue_id:dialogueId})});
      const data = await response.json();
      if (!response.ok) throw Error(data.error || 'Check file failed');
      if (checkToken !== offlineRunToken) return;
      checked = data.result; $('offlinePredict').disabled = false; latestPredictions = [];
      $('rows').innerHTML = '<tr><td colspan="8" class="muted">   </td></tr>';
      $('offlineSummary').innerHTML = '';
      setStatus(`Đã khớp ${checked.utterance_count} utterances · ${checked.dataset} ${checked.dialogue_id}`);
    } catch (error) { if (checkToken === offlineRunToken) setStatus(error.message, true); }
    finally { if (checkToken === offlineRunToken) $('checkFile').disabled = false; }
  });
  $('offlinePredict').addEventListener('click', async () => {
    if (!checked) return setStatus('Hãy bấm Check file trước.', true);
    try {
      $('offlinePredict').disabled = true;
      setStatus(`Đang chạy ${$('model').value} · ${checked.dataset}…`);
      const response = await fetch('/api/offline/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({dataset:checked.dataset, model:$('model').value, dialogue_id:checked.dialogue_id})});
      const data = await response.json();
      if (!response.ok) throw Error(data.error || 'Prediction failed');
      const allResults = data.results || [];
      const token = ++offlineRunToken;
      const revealed = [];
      for (let index = 0; index < allResults.length; index++) {
        if (token !== offlineRunToken) return;
        revealed.push(allResults[index]);
        latestPredictions = revealed.slice();
        const byId = Object.fromEntries(revealed.map(x => [String(x.utterance_id), x]));
        renderRows(checked.rows, byId);
        setStatus(`${$('model').value}: ${index + 1}/${allResults.length} utterances`);
        if (index < allResults.length - 1) await new Promise(resolve => setTimeout(resolve, 2000));
      }
      const modelName = $('model').value;
      offlineComparisons[checked.dataset][modelName] = buildManualRun(modelName, checked.rows, allResults);
      renderOfflineModelSummary();
      $('downloadCsv').disabled = latestPredictions.length === 0;
      $('offlinePredict').disabled = false;
      setStatus(`Hoàn tất ${data.count} predictions · ${modelName}`);
    } catch (error) { $('offlinePredict').disabled = false; setStatus(error.message, true); }
  });

  $('downloadCsv').addEventListener('click', () => {
    const rows = [['model','dataset','dialogue_id','utterance_id','speaker','start','end','text','emotion','confidence'], ...latestPredictions.map(x => [x.model,x.dataset,x.dialogue_id,x.utterance_id,x.speaker,x.start_time,x.end_time,x.text,x.emotion,x.confidence])];
    const csv = rows.map(row => row.map(value => `"${String(value ?? '').replaceAll('"','""')}"`).join(',')).join('\\n');
    const blobUrl = URL.createObjectURL(new Blob([csv], {type:'text/csv'}));
    const link = document.createElement('a'); link.href = blobUrl; link.download = 'offline_predictions.csv'; link.click();
    window.setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
  });

  function renderRows(rows, predictions = {}) {
    const revealedIds = new Set(Object.keys(predictions));
    const visibleRows = revealedIds.size ? rows.filter(row => revealedIds.has(String(row.utterance_id))) : rows;
    let correct = 0; let scored = 0;
    $('rows').innerHTML = visibleRows.map((row, index) => {
      const pred = predictions[String(row.utterance_id)];
      const truth = String(row.emotion || '').toLowerCase();
      const predicted = String(pred?.emotion || '').toLowerCase();
      const hasScore = Boolean(pred && truth);
      const isCorrect = hasScore && truth === predicted;
      if (hasScore) { scored += 1; if (isCorrect) correct += 1; }
      const predClass = hasScore ? (isCorrect ? 'correct' : 'wrong') : '';
      return `<tr><td><span class="data-box">${index + 1}</span></td><td><span class="data-box">${esc(row.speaker)}</span></td><td><span class="data-box">${Number(row.start).toFixed(2)}s</span></td><td><span class="data-box">${Number(row.end).toFixed(2)}s</span></td><td><span class="compact-text" title="${esc(row.text)}">${esc(row.text)}</span></td><td><span class="data-box">${esc(row.emotion || '—')}</span></td><td><span class="data-box prediction-box ${predClass}">${pred ? `${esc(pred.emotion)} ${(Number(pred.confidence) * 100).toFixed(1)}%` : '—'}</span></td><td><button class="check-btn offline-jump" data-time="${Number(row.start)}">Xem đoạn</button></td></tr>`;
    }).join('');
    document.querySelectorAll('.offline-jump').forEach(button => button.addEventListener('click', () => { const player = $('offlineVideo'); player.currentTime = Number(button.dataset.time); player.play().catch(() => {}); }));
    $('offlineSummary').innerHTML = '';
  }

  function renderSummary(total, scored, correct) {
    const accuracy = scored ? (correct / scored * 100) : 0;
    $('offlineSummary').innerHTML = `<div class="summary-box"><strong>${total}</strong><span>Tổng câu</span></div><div class="summary-box"><strong>${correct}</strong><span>Đúng nhãn</span></div><div class="summary-box"><strong>${Math.max(0, scored - correct)}</strong><span>Sai nhãn</span></div><div class="summary-box"><strong>${accuracy.toFixed(1)}%</strong><span>Accuracy</span></div>`;
  }
  function buildManualRun(model, rows, results) {
    const truth = Object.fromEntries(rows.map(row => [String(row.utterance_id), String(row.emotion || '').toLowerCase()]));
    let correct = 0; let total = 0;
    for (const result of results) {
      const label = truth[String(result.utterance_id)];
      if (!label) continue;
      total += 1;
      if (String(result.emotion || '').toLowerCase() === label) correct += 1;
    }
    return { model, results: results.slice(), metrics: {total, correct, wrong: total - correct, accuracy: total ? correct / total : 0} };
  }
  function renderOfflineModelSummary() {
    const datasetCard = dataset => {
      const data = offlineComparisons[dataset];
      const cards = ['CRG-3','DiGemo','DMCER'].map(model => {
        const run = data[model];
        if (!run) return `<div class="model-metric empty"><strong>${esc(model)}</strong><span>Chưa chạy</span></div>`;
        const m = run.metrics; const pct = Number(m.accuracy || 0) * 100;
        return `<div class="model-metric"><strong>${esc(model)}</strong><span>${m.correct}/${m.total} đúng</span><span>${m.wrong} sai</span><b>${pct.toFixed(1)}%</b><i class="accuracy-bar"><span style="width:${Math.max(0, Math.min(100, pct))}%"></span></i></div>`;
      }).join('');
      return `<section class="model-compare"><h3>${dataset}</h3><div class="model-metrics">${cards}</div></section>`;
    };
    $('offlineSummary').innerHTML = `<div class="compare-grid">${datasetCard('MELD')}${datasetCard('IEMOCAP')}</div>`;
  }

})();