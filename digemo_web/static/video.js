(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const names = {neutral:'Neutral',surprise:'Surprise',fear:'Fear',sadness:'Sadness',joy:'Joy',disgust:'Disgust',anger:'Anger'};
  const faces = {neutral:'😐',surprise:'😮',fear:'😨',sadness:'😔',joy:'😊',disgust:'😖',anger:'😠'};
  const state = {token:null, rows:[], slots:{}, nSlots:9, results:null, resultCsv:'', busy:false, videoUrl:null, stopAt:null, meld:false, meldDia:null};

  const toast = (msg) => { $('toastText').textContent = msg; $('toast').classList.remove('hidden'); };
  $('toastClose').addEventListener('click', () => $('toast').classList.add('hidden'));
  const api = async (path, options = {}) => {
    const r = await fetch(path, options);
    let data;
    try { data = await r.json(); } catch (_) { throw new Error(`HTTP ${r.status}: máy chủ không phản hồi JSON.`); }
    if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${r.status}`);
    return data;
  };
  const fmt = (s) => { s = Math.max(0, +s || 0); const m = Math.floor(s / 60); return `${m}:${(s - m * 60).toFixed(2).padStart(5, '0')}`; };

  function setBusy(flag, tag) {
    state.busy = flag;
    $('analyzeBtn').disabled = flag || !$('videoFile').files[0];
    $('clearBtn').disabled = flag || !state.token;
    $('predictBtn').disabled = flag;
    $('addRowBtn').disabled = flag;
    $('jobTag').textContent = tag || (flag ? 'ĐANG XỬ LÝ' : 'SẴN SÀNG');
    $('jobTag').className = 'tag ' + (flag ? '' : 'ready');
  }
  function showProgress(p, msg) {
    $('progress').classList.remove('hidden');
    $('progressBar').style.width = `${Math.round(p * 100)}%`;
    $('progressText').textContent = `${Math.round(p * 100)}% · ${msg || ''}`;
  }
  async function pollJob(id) {
    for (;;) {
      const j = await api(`/api/video/jobs/${id}`);
      showProgress(j.progress, j.message);
      if (j.status === 'done') return j.result;
      if (j.status === 'error') throw new Error(j.error || 'Tác vụ thất bại.');
      await new Promise(r => setTimeout(r, 900));
    }
  }

  /* ---------- step 1 ---------- */
  function pickFile(inputId, nameId, emptyText) {
    const f = $(inputId).files[0];
    $(nameId).textContent = f ? f.name : emptyText;
    $(nameId).closest('.drop').classList.toggle('has-file', !!f);
  }
  $('videoFile').addEventListener('change', () => {
    pickFile('videoFile', 'videoName', 'Chọn video (mp4, avi, mov, mkv, webm)');
    $('analyzeBtn').disabled = state.busy || !$('videoFile').files[0];
  });
  $('csvFile').addEventListener('change', () => {
    pickFile('csvFile', 'csvName', 'CSV utterance (tùy chọn)');
    $('autoHint').textContent = $('csvFile').files[0]
      ? 'Có CSV: dùng đúng Speaker, thời gian và Dialogue_ID trong file. Dòng thiếu Utterance sẽ được Whisper nhận dạng.'
      : 'Không có CSV: Whisper tự tách câu và phân cụm giọng thành N người. Cách này gần đúng, hãy kiểm tra và sửa cột Speaker ở bước 2.';
  });
  const setN = (n) => {
    $('numSpeakers').value = n;
    document.querySelectorAll('#speakerChips .chip').forEach(c => c.classList.toggle('active', +c.dataset.n === +n));
  };
  document.querySelectorAll('#speakerChips .chip').forEach(c => c.addEventListener('click', () => setN(c.dataset.n)));
  $('numSpeakers').addEventListener('input', () => setN($('numSpeakers').value));

  async function analyze() {
    const file = $('videoFile').files[0];
    if (!file || state.busy) return;
    const n = Math.round(+$('numSpeakers').value);
    if (!(n >= 1 && n <= 8)) { toast('Số người nói phải từ 1 đến 8.'); return; }
    await clearVideo(true);
    const form = new FormData();
    form.append('video', file, file.name);
    const csv = $('csvFile').files[0];
    if (csv) form.append('csv_file', csv, csv.name);
    form.append('num_speakers', String(n));
    if ($('meldMode').checked) {
      const id = parseInt($('meldId').value, 10);
      if (!Number.isInteger(id) || id < 0) { toast('Chế độ MELD: hãy nhập Dialogue_ID hợp lệ.'); return; }
      form.append('meld_dialogue', String(id));
      form.append('meld_time_mode', $('meldTime').value);
    }
    setBusy(true);
    try {
      showProgress(0.01, 'Đang tải video lên…');
      const start = await api('/api/video/analyze', {method:'POST', body:form});
      state.token = start.token;
      const res = await pollJob(start.job_id);
      state.rows = res.rows.map(r => ({...r}));
      state.slots = res.slots;
      state.nSlots = res.n_slots || 9;
      state.results = null;
      state.meld = res.source === 'meld';
      state.meldDia = state.meld ? res.meld_dialogue : null;
      if (res.warning) toast(res.warning);
      if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
      state.videoUrl = URL.createObjectURL(file);
      $('player').src = state.videoUrl;
      $('reviewGrid').classList.remove('hidden');
      $('tableCard').classList.remove('hidden');
      renderAll();
      setBusy(false, 'ĐÃ PHÂN ĐOẠN');
    } catch (e) {
      toast(e.message);
      if (state.token) { fetch(`/api/video/${state.token}`, {method:'DELETE'}).catch(() => {}); state.token = null; }
      setBusy(false, 'LỖI');
    }
  }
  async function clearVideo(silent) {
    if (state.token) await fetch(`/api/video/${state.token}`, {method:'DELETE'}).catch(() => {});
    state.token = null; state.rows = []; state.results = null; state.resultCsv = '';
    state.meld = false; state.meldDia = null;
    $('reportCard').classList.add('hidden');
    $('reviewGrid').classList.add('hidden'); $('tableCard').classList.add('hidden');
    $('progress').classList.add('hidden');
    if (!silent) setBusy(false, 'CHƯA CHẠY');
  }
  $('analyzeBtn').addEventListener('click', analyze);
  $('clearBtn').addEventListener('click', () => clearVideo(false));

  /* ---------- step 2: table ---------- */
  const speakerNames = () => [...new Set(state.rows.map(r => (r.speaker || '').trim()).filter(Boolean))];

  function renderSlots() {
    const list = $('slotList');
    const used = new Set();
    const items = speakerNames().map(s => {
      let v = state.slots[s];
      if (v === undefined || v === null) { v = [...Array(state.nSlots).keys()].find(i => !used.has(i)) ?? state.nSlots - 1; state.slots[s] = v; }
      used.add(v);
      const row = document.createElement('div'); row.className = 'slot-item';
      const label = document.createElement('span'); label.textContent = s;
      const sel = document.createElement('select');
      for (let i = 0; i < state.nSlots; i++) sel.add(new Option(i === state.nSlots - 1 ? `Slot ${i} (Other)` : `Slot ${i}`, i));
      sel.value = v;
      sel.addEventListener('change', () => { state.slots[s] = +sel.value; });
      row.append(label, sel);
      return row;
    });
    list.replaceChildren(...items);
    $('speakerList').replaceChildren(...speakerNames().map(s => new Option(s)));
  }

  function mkInput(value, type, onInput, extra = {}) {
    const el = document.createElement('input');
    el.type = type; el.value = value;
    Object.assign(el, extra);
    el.addEventListener('input', () => onInput(el.value));
    return el;
  }

  function renderTable() {
    const table = $('uttTable');
    const head = document.createElement('div'); head.className = 'utt-row head';
    table.classList.toggle('meld', state.meld);
    const heads = state.meld
      ? ['#', 'SPEAKER', 'BẮT ĐẦU (s)', 'KẾT THÚC (s)', 'TRANSCRIPT', 'MODEL DỰ ĐOÁN', 'NHÃN CSV', '', '']
      : ['#', 'SPEAKER', 'BẮT ĐẦU (s)', 'KẾT THÚC (s)', 'TRANSCRIPT', 'CẢM XÚC', '', ''];
    heads.forEach(t => {
      const s = document.createElement('span'); s.textContent = t; head.append(s);
    });
    const byIdx = new Map((state.results ? state.results.results : []).map(r => [r.src_index, r]));
    const rows = state.rows.map((r, i) => {
      const row = document.createElement('div'); row.className = 'utt-row'; row.dataset.i = i;
      const num = document.createElement('span'); num.className = 'num'; num.textContent = String(i + 1).padStart(2, '0');
      const spk = mkInput(r.speaker, 'text', v => { r.speaker = v; });
      spk.setAttribute('list', 'speakerList');
      spk.addEventListener('change', renderSlots);
      const st = mkInput(r.start, 'number', v => { r.start = parseFloat(v); }, {step: 0.01, min: 0});
      const en = mkInput(r.end, 'number', v => { r.end = parseFloat(v); }, {step: 0.01, min: 0});
      const tx = mkInput(r.text, 'text', v => { r.text = v; }, {maxLength: 1500});
      const res = document.createElement('div'); res.className = 'res';
      const hit = byIdx.get(i);
      if (hit) {
        const chip = document.createElement('div'); chip.className = 'emotion-chip';
        chip.textContent = `${faces[hit.emotion] || '•'} ${names[hit.emotion] || hit.emotion}`;
        const small = document.createElement('small');
        const top3 = [...hit.probabilities].sort((a, b) => b.percent - a.percent).slice(0, 3)
          .map(p => `${names[p.label] || p.label} ${p.percent.toFixed(0)}%`).join(' · ');
        small.textContent = `${hit.confidence.toFixed(1)}% · ${top3}` + (hit.truth ? ` · đúng: ${names[hit.truth] || hit.truth}` : '');
        res.append(chip, small);
        if (state.meld) {
          res.classList.add('model');
          small.textContent = `${hit.confidence.toFixed(1)}%`;
          if (hit.correct !== null && hit.correct !== undefined) {
            const v = document.createElement('div'); v.className = 'verdict';
            v.textContent = hit.correct ? '✓ ĐÚNG' : '✗ SAI';
            res.append(v);
          }
        }
        if (hit.correct === true) row.classList.add('right');
        if (hit.correct === false) row.classList.add('wrong');
      }
      const csvRes = document.createElement('div'); csvRes.className = 'res csvres';
      if (state.meld) {
        const c = document.createElement('div'); c.className = 'emotion-chip';
        c.textContent = r.truth ? `${faces[r.truth] || '•'} ${names[r.truth] || r.truth}` : '—';
        csvRes.append(c);
      }
      const play = document.createElement('button'); play.type = 'button'; play.className = 'mini';
      play.innerHTML = '<svg class="icon"><use href="#i-play"/></svg>';
      play.setAttribute('aria-label', 'Phát đoạn này');
      play.addEventListener('click', () => playRow(i));
      const del = document.createElement('button'); del.type = 'button'; del.className = 'mini';
      del.innerHTML = '<svg class="icon"><use href="#i-trash"/></svg>';
      del.setAttribute('aria-label', 'Xóa dòng');
      del.addEventListener('click', () => { state.rows.splice(i, 1); state.results = null; renderAll(); });
      if (state.meld) row.append(num, spk, st, en, tx, res, csvRes, play, del);
      else row.append(num, spk, st, en, tx, res, play, del);
      return row;
    });
    table.replaceChildren(head, ...rows);
    $('rowCount').textContent = state.rows.length;
  }

  function renderSummary() {
    const box = $('summary');
    const r = state.results;
    $('dlResultBtn').classList.toggle('hidden', !r);
    if (!r) { box.classList.add('hidden'); box.replaceChildren(); return; }
    const pills = Object.entries(r.counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => {
      const p = document.createElement('span'); p.className = 'pill'; p.textContent = `${faces[k] || ''} ${names[k] || k}: ${v}`; return p;
    });
    if (r.accuracy !== null && r.accuracy !== undefined) {
      const a = document.createElement('span'); a.className = 'pill good';
      a.textContent = `Độ chính xác so với cột Emotion: ${r.accuracy}% (${r.graded} câu có nhãn)`; pills.unshift(a);
    }
    box.replaceChildren(...pills); box.classList.remove('hidden');
  }

  function renderAll() { renderSlots(); renderTable(); renderSummary(); renderReport(); }

  $('addRowBtn').addEventListener('click', () => {
    const last = state.rows[state.rows.length - 1];
    const s = last ? last.end : 0;
    state.rows.push({start: s, end: s + 2, speaker: last ? last.speaker : 'Speaker 1', text: '', dialogue_id: null, truth: null});
    state.results = null; renderAll();
  });

  function playRow(i) {
    const r = state.rows[i], v = $('player');
    if (!v.src) return;
    v.currentTime = r.start; state.stopAt = r.end; v.play().catch(() => {});
    document.querySelectorAll('.utt-row').forEach(el => el.classList.toggle('playing', +el.dataset.i === i));
  }
  $('player').addEventListener('timeupdate', () => {
    const v = $('player');
    if (state.stopAt !== null && v.currentTime >= state.stopAt) { v.pause(); state.stopAt = null; document.querySelectorAll('.utt-row.playing').forEach(el => el.classList.remove('playing')); }
  });
  $('player').addEventListener('seeking', () => { /* manual seek cancels the auto-stop only when far from the row */ });

  /* ---------- CSV ---------- */
  const csvCell = (v) => { const s = String(v ?? ''); return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  function download(name, text) {
    const blob = new Blob(['\ufeff' + text], {type: 'text/csv;charset=utf-8'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  $('dlInputBtn').addEventListener('click', () => {
    const rows = [...state.rows].sort((a, b) => a.start - b.start);
    const lines = [['Dialogue_ID', 'Utterance_ID', 'Speaker', 'Utterance', 'StartTime', 'EndTime', 'Emotion'].join(',')];
    rows.forEach((r, i) => lines.push([r.dialogue_id ?? 0, i, r.speaker, r.text, r.start, r.end, r.truth ?? ''].map(csvCell).join(',')));
    download('utterances_input.csv', lines.join('\r\n'));
  });
  $('dlResultBtn').addEventListener('click', () => { if (state.resultCsv) download('digemo_predictions.csv', state.resultCsv); });

  /* ---------- predict ---------- */
  async function predict() {
    if (state.busy || !state.token) return;
    const bad = state.rows.findIndex(r => !(r.text || '').trim() || !(r.speaker || '').trim() || !(r.end > r.start));
    if (bad >= 0) { toast(`Dòng ${bad + 1}: cần speaker, transcript và thời gian kết thúc lớn hơn bắt đầu.`); return; }
    state.rows.sort((a, b) => a.start - b.start);
    renderAll();
    const body = {
      token: state.token,
      rows: state.rows.map(r => ({start: r.start, end: r.end, speaker: r.speaker.trim(), text: r.text.trim(), dialogue_id: r.dialogue_id ?? null, truth: r.truth ?? null})),
      slots: state.slots,
      max_per_dialogue: state.meld ? 100 : (Math.round(+$('maxPer').value) || 30),
      face_crop: $('faceCrop').checked,
    };
    setBusy(true);
    try {
      showProgress(0.01, 'Đang gửi yêu cầu…');
      const start = await api('/api/video/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
      const res = await pollJob(start.job_id);
      state.results = res; state.resultCsv = res.csv; state.slots = {...state.slots, ...res.slots};
      if (state.meld) recordRun(res);
      renderAll();
      setBusy(false, 'XONG');
    } catch (e) { toast(e.message); setBusy(false, 'LỖI'); }
  }
  $('predictBtn').addEventListener('click', predict);

  /* ---------- MELD mode: toggle + preview ---------- */
  const EMO = Object.keys(names);
  const short = (l) => (names[l] || l).slice(0, 3).toUpperCase();
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };

  function syncMeldUi() {
    const on = $('meldMode').checked;
    $('meldBox').classList.toggle('hidden', !on);
    $('csvDrop').classList.toggle('hidden', on);
    $('speakerOpt').classList.toggle('hidden', on);
    $('autoHint').classList.toggle('hidden', on);
  }
  $('meldMode').addEventListener('change', syncMeldUi);

  async function previewMeld() {
    const id = parseInt($('meldId').value, 10);
    if (!Number.isInteger(id) || id < 0) { toast('Hãy nhập Dialogue_ID hợp lệ (số nguyên ≥ 0).'); return; }
    const box = $('meldPreview');
    try {
      const d = await api(`/api/video/meld/${id}?time_mode=${encodeURIComponent($('meldTime').value)}`);
      const counts = {};
      d.rows.forEach(r => { if (r.truth) counts[r.truth] = (counts[r.truth] || 0) + 1; });
      const pills = [`Dialogue ${d.dialogue_id}`, `${d.count} câu`, `${d.duration}s`, `Người nói: ${d.speakers.join(', ')}`].map(t => el('span', 'pill', t));
      Object.entries(counts).forEach(([k, v]) => pills.push(el('span', 'pill', `${faces[k] || ''} ${names[k] || k}: ${v}`)));
      box.replaceChildren(...pills); box.classList.remove('hidden');
    } catch (e) { box.classList.add('hidden'); toast(e.message); }
  }
  $('meldLoadBtn').addEventListener('click', previewMeld);

  /* ---------- run history (localStorage, latest run per dialogue) ---------- */
  const RUNS_KEY = 'digemo_meld_runs';
  const loadRuns = () => { try { return JSON.parse(localStorage.getItem(RUNS_KEY) || '[]'); } catch (_) { return []; } };
  const saveRuns = (runs) => { try { localStorage.setItem(RUNS_KEY, JSON.stringify(runs.slice(-40))); } catch (_) {} };
  function recordRun(res) {
    if (res.graded <= 0) return;
    const correct = res.results.filter(r => r.correct === true).length;
    const runs = loadRuns().filter(r => r.dia !== state.meldDia);
    runs.push({dia: state.meldDia, correct, graded: res.graded, ts: Date.now()});
    saveRuns(runs);
  }
  $('clearRunsBtn').addEventListener('click', () => { saveRuns([]); renderReport(); });

  /* ---------- report charts ---------- */
  function renderReport() {
    const card = $('reportCard');
    const res = state.results;
    if (!state.meld || !res) { card.classList.add('hidden'); return; }
    const items = res.results;
    const graded = items.filter(r => r.truth);
    const ok = graded.filter(r => r.correct === true).length;
    const bad = graded.length - ok;
    const acc = graded.length ? ok / graded.length : 0;
    card.classList.remove('hidden');
    $('reportDia').textContent = `#${state.meldDia}`;
    $('reportTag').textContent = graded.length ? `${ok}/${graded.length} CÂU ĐÚNG` : 'KHÔNG CÓ NHÃN';

    // donut: red base ring, green arc on top
    const C = 2 * Math.PI * 48;
    const ns = 'http://www.w3.org/2000/svg';
    const ring = (stroke, dash) => {
      const c = document.createElementNS(ns, 'circle');
      c.setAttribute('cx', 60); c.setAttribute('cy', 60); c.setAttribute('r', 48);
      c.setAttribute('fill', 'none'); c.setAttribute('stroke', stroke); c.setAttribute('stroke-width', 14);
      if (dash !== undefined) { c.setAttribute('stroke-dasharray', `${dash} ${C}`); c.setAttribute('stroke-linecap', 'round'); }
      c.setAttribute('transform', 'rotate(-90 60 60)');
      return c;
    };
    $('donut').replaceChildren(
      ring(graded.length ? '#ff6f86' : '#28374b'),
      ...(graded.length && ok > 0 ? [ring('#5be4b7', Math.max(0.1, acc * C - (ok < graded.length ? 3 : 0)))] : []));
    $('donutPct').textContent = graded.length ? `${(acc * 100).toFixed(0)}%` : '—';
    $('reportStats').replaceChildren(
      ...[['Câu có nhãn', graded.length, ''], ['Dự đoán đúng', ok, 'good'], ['Dự đoán sai', bad, 'bad']].map(([t, v, c]) => {
        const s = el('div', `stat ${c}`); s.append(el('small', '', t), el('strong', '', String(v))); return s;
      }));

    // timeline
    $('timeline').replaceChildren(...items.map((r, i) => {
      const cls = !r.truth ? 'na' : r.correct ? 'ok' : 'no';
      const c = el('div', `tl-cell ${cls}`);
      c.append(el('span', '', !r.truth ? '·' : r.correct ? '✓' : '✗'), document.createTextNode(String(i + 1)));
      c.title = `#${i + 1} ${r.speaker}: model = ${names[r.emotion] || r.emotion}` + (r.truth ? ` · CSV = ${names[r.truth] || r.truth}` : '');
      return c;
    }));

    // per-emotion stacked bars (by true label)
    const per = Object.fromEntries(EMO.map(l => [l, {ok: 0, no: 0}]));
    graded.forEach(r => { if (per[r.truth]) per[r.truth][r.correct ? 'ok' : 'no']++; });
    $('emoBars').replaceChildren(...EMO.map(l => {
      const {ok: a, no: b} = per[l], n = a + b;
      const row = el('div', 'emo-row'); row.append(el('span', 'nm', `${faces[l]} ${names[l]}`));
      const track = el('div', 'emo-track');
      if (n) {
        const g = el('i', 'g'); g.style.width = `${a / n * 100}%`;
        const r = el('i', 'r'); r.style.width = `${b / n * 100}%`;
        track.append(g, r);
      }
      row.append(track, el('span', 'ct', n ? `${a}/${n}` : '—'));
      return row;
    }));

    // confusion matrix (rows = CSV truth, cols = model)
    const idx = Object.fromEntries(EMO.map((l, i) => [l, i]));
    const M = EMO.map(() => EMO.map(() => 0));
    graded.forEach(r => { if (r.truth in idx && r.emotion in idx) M[idx[r.truth]][idx[r.emotion]]++; });
    const cells = [el('span', 'h', 'THẬT↓')];
    EMO.forEach(l => cells.push(el('span', 'h', short(l))));
    M.forEach((rowv, i) => {
      const tot = rowv.reduce((a, b) => a + b, 0);
      cells.push(el('span', 'h', short(EMO[i])));
      rowv.forEach((v, j) => {
        const c = el('span', `c${i === j ? ' diag' : ''}`, v ? String(v) : '');
        if (v) {
          const f = v / tot;
          c.style.background = i === j ? `rgba(101,230,216,${0.25 + 0.75 * f})` : `rgba(255,111,134,${0.25 + 0.75 * f})`;
          if (f > 0.5) c.classList.add('hit');
        }
        c.title = `Thật: ${names[EMO[i]]} → Model: ${names[EMO[j]]} (${v})`;
        cells.push(c);
      });
    });
    const conf = el('div', 'conf'); conf.append(...cells);
    $('confMatrix').replaceChildren(conf);

    // run history
    const runs = loadRuns();
    const list = $('runList');
    if (!runs.length) { list.replaceChildren(el('div', 'run-empty', 'Chưa có dialogue nào được ghi lại.')); $('runTotal').textContent = ''; }
    else {
      list.replaceChildren(...runs.map(r => {
        const row = el('div', 'run-item');
        const bar = el('div', 'bar'), fill = el('i'); fill.style.width = `${r.correct / r.graded * 100}%`; bar.append(fill);
        row.append(el('span', '', `Dialogue ${r.dia}`), bar, el('span', 'meta', `${r.correct}/${r.graded} · ${(r.correct / r.graded * 100).toFixed(0)}%`));
        return row;
      }));
      const C2 = runs.reduce((a, r) => a + r.correct, 0), G2 = runs.reduce((a, r) => a + r.graded, 0);
      $('runTotal').textContent = `TỔNG ${runs.length} DIALOGUE: ${C2}/${G2} = ${(C2 / G2 * 100).toFixed(1)}%`;
    }
  }

  window.addEventListener('beforeunload', () => { if (state.token) navigator.sendBeacon?.(`/api/video/${state.token}/delete`); });
  setBusy(false, 'CHƯA CHẠY');
})();