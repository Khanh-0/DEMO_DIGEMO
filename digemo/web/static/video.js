// /* DiGemo video analysis page: upload -> editable utterance table -> prediction. */
// (() => {
//   'use strict';
//   const $ = (id) => document.getElementById(id);
//   const names = {neutral:'Neutral',surprise:'Surprise',fear:'Fear',sadness:'Sadness',joy:'Joy',disgust:'Disgust',anger:'Anger'};
//   const faces = {neutral:'😐',surprise:'😮',fear:'😨',sadness:'😔',joy:'😊',disgust:'😖',anger:'😠'};
//   const state = {token:null, rows:[], slots:{}, nSlots:9, results:null, resultCsv:'', busy:false, videoUrl:null, stopAt:null, meld:false, meldDia:null};

//   const toast = (msg) => { $('toastText').textContent = msg; $('toast').classList.remove('hidden'); };
//   $('toastClose').addEventListener('click', () => $('toast').classList.add('hidden'));
//   const api = async (path, options = {}) => {
//     const r = await fetch(path, options);
//     let data;
//     try { data = await r.json(); } catch (_) { throw new Error(`HTTP ${r.status}: máy chủ không phản hồi JSON.`); }
//     if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${r.status}`);
//     return data;
//   };
//   const fmt = (s) => { s = Math.max(0, +s || 0); const m = Math.floor(s / 60); return `${m}:${(s - m * 60).toFixed(2).padStart(5, '0')}`; };

//   function setBusy(flag, tag) {
//     state.busy = flag;
//     $('analyzeBtn').disabled = flag || !$('videoFile').files[0];
//     $('clearBtn').disabled = flag || !state.token;
//     $('predictBtn').disabled = flag;
//     $('addRowBtn').disabled = flag;
//     $('jobTag').textContent = tag || (flag ? 'ĐANG XỬ LÝ' : 'SẴN SÀNG');
//     $('jobTag').className = 'tag ' + (flag ? '' : 'ready');
//   }
//   function showProgress(p, msg) {
//     $('progress').classList.remove('hidden');
//     $('progressBar').style.width = `${Math.round(p * 100)}%`;
//     $('progressText').textContent = `${Math.round(p * 100)}% · ${msg || ''}`;
//   }
//   async function pollJob(id) {
//     for (;;) {
//       const j = await api(`/api/video/jobs/${id}`);
//       showProgress(j.progress, j.message);
//       if (j.status === 'done') return j.result;
//       if (j.status === 'error') throw new Error(j.error || 'Tác vụ thất bại.');
//       await new Promise(r => setTimeout(r, 900));
//     }
//   }

//   /* ---------- step 1 ---------- */
//   function pickFile(inputId, nameId, emptyText) {
//     const f = $(inputId).files[0];
//     $(nameId).textContent = f ? f.name : emptyText;
//     $(nameId).closest('.drop').classList.toggle('has-file', !!f);
//   }
//   $('videoFile').addEventListener('change', () => {
//     pickFile('videoFile', 'videoName', 'Chọn video (mp4, avi, mov, mkv, webm)');
//     $('analyzeBtn').disabled = state.busy || !$('videoFile').files[0];
//   });
//   $('csvFile').addEventListener('change', () => {
//     pickFile('csvFile', 'csvName', 'CSV utterance (tùy chọn)');
//     $('autoHint').textContent = $('csvFile').files[0]
//       ? 'Có CSV: dùng đúng Speaker, thời gian và Dialogue_ID trong file. Dòng thiếu Utterance sẽ được Whisper nhận dạng.'
//       : 'Không có CSV: Whisper tự tách câu và phân cụm giọng thành N người. Cách này gần đúng, hãy kiểm tra và sửa cột Speaker ở bước 2.';
//   });
//   const setN = (n) => {
//     $('numSpeakers').value = n;
//     document.querySelectorAll('#speakerChips .chip').forEach(c => c.classList.toggle('active', +c.dataset.n === +n));
//   };
//   document.querySelectorAll('#speakerChips .chip').forEach(c => c.addEventListener('click', () => setN(c.dataset.n)));
//   $('numSpeakers').addEventListener('input', () => setN($('numSpeakers').value));

//   async function analyze() {
//     const file = $('videoFile').files[0];
//     if (!file || state.busy) return;
//     const n = Math.round(+$('numSpeakers').value);
//     if (!(n >= 1 && n <= 8)) { toast('Số người nói phải từ 1 đến 8.'); return; }
//     await clearVideo(true);
//     const form = new FormData();
//     form.append('video', file, file.name);
//     const csv = $('csvFile').files[0];
//     if (csv) form.append('csv_file', csv, csv.name);
//     form.append('num_speakers', String(n));
//     if ($('meldMode').checked) {
//       const id = parseInt($('meldId').value, 10);
//       if (!Number.isInteger(id) || id < 0) { toast('Chế độ MELD: hãy nhập Dialogue_ID hợp lệ.'); return; }
//       form.append('meld_dialogue', String(id));
//       form.append('meld_time_mode', $('meldTime').value);
//     }
//     setBusy(true);
//     try {
//       showProgress(0.01, 'Đang tải video lên…');
//       const start = await api('/api/video/analyze', {method:'POST', body:form});
//       state.token = start.token;
//       const res = await pollJob(start.job_id);
//       state.rows = res.rows.map(r => ({...r}));
//       state.slots = res.slots;
//       state.nSlots = res.n_slots || 9;
//       state.results = null;
//       state.meld = res.source === 'meld';
//       state.meldDia = state.meld ? res.meld_dialogue : null;
//       if (res.warning) toast(res.warning);
//       if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
//       state.videoUrl = URL.createObjectURL(file);
//       $('player').src = state.videoUrl;
//       $('reviewGrid').classList.remove('hidden');
//       $('tableCard').classList.remove('hidden');
//       renderAll();
//       setBusy(false, 'ĐÃ PHÂN ĐOẠN');
//       if (window.updateStepper) window.updateStepper(2);
//     } catch (e) {
//       toast(e.message);
//       if (state.token) { fetch(`/api/video/${state.token}`, {method:'DELETE'}).catch(() => {}); state.token = null; }
//       setBusy(false, 'LỖI');
//     }
//   }
//   async function clearVideo(silent) {
//     if (state.token) await fetch(`/api/video/${state.token}`, {method:'DELETE'}).catch(() => {});
//     state.token = null; state.rows = []; state.results = null; state.resultCsv = '';
//     state.meld = false; state.meldDia = null;
//     $('reportCard').classList.add('hidden');
//     $('reviewGrid').classList.add('hidden'); $('tableCard').classList.add('hidden');
//     $('progress').classList.add('hidden');
//     if (!silent) setBusy(false, 'CHƯA CHẠY');
//     if (!silent && window.updateStepper) window.updateStepper(1);
//   }
//   $('analyzeBtn').addEventListener('click', analyze);
//   $('clearBtn').addEventListener('click', () => clearVideo(false));

//   /* ---------- step 2: table ---------- */
//   const speakerNames = () => [...new Set(state.rows.map(r => (r.speaker || '').trim()).filter(Boolean))];

//   function renderSlots() {
//     const list = $('slotList');
//     const used = new Set();
//     const items = speakerNames().map(s => {
//       let v = state.slots[s];
//       if (v === undefined || v === null) { v = [...Array(state.nSlots).keys()].find(i => !used.has(i)) ?? state.nSlots - 1; state.slots[s] = v; }
//       used.add(v);
//       const row = document.createElement('div'); row.className = 'slot-item';
//       const label = document.createElement('span'); label.textContent = s;
//       const sel = document.createElement('select');
//       for (let i = 0; i < state.nSlots; i++) sel.add(new Option(i === state.nSlots - 1 ? `Slot ${i} (Other)` : `Slot ${i}`, i));
//       sel.value = v;
//       sel.addEventListener('change', () => { state.slots[s] = +sel.value; });
//       row.append(label, sel);
//       return row;
//     });
//     list.replaceChildren(...items);
//     $('speakerList').replaceChildren(...speakerNames().map(s => new Option(s)));
//   }

//   function mkInput(value, type, onInput, extra = {}) {
//     const el = document.createElement('input');
//     el.type = type; el.value = value;
//     Object.assign(el, extra);
//     el.addEventListener('input', () => onInput(el.value));
//     return el;
//   }

//   function renderTable() {
//     const table = $('uttTable');
//     const head = document.createElement('div'); head.className = 'utt-row head';
//     table.classList.toggle('meld', state.meld);
//     const heads = state.meld
//       ? ['#', 'SPEAKER', 'BẮT ĐẦU (s)', 'KẾT THÚC (s)', 'TRANSCRIPT', 'MODEL DỰ ĐOÁN', 'NHÃN CSV', '', '']
//       : ['#', 'SPEAKER', 'BẮT ĐẦU (s)', 'KẾT THÚC (s)', 'TRANSCRIPT', 'CẢM XÚC', '', ''];
//     heads.forEach(t => {
//       const s = document.createElement('span'); s.textContent = t; head.append(s);
//     });
//     const byIdx = new Map((state.results ? state.results.results : []).map(r => [r.src_index, r]));
//     const rows = state.rows.map((r, i) => {
//       const row = document.createElement('div'); row.className = 'utt-row'; row.dataset.i = i;
//       const num = document.createElement('span'); num.className = 'num'; num.textContent = String(i + 1).padStart(2, '0');
//       if ((r.overlap || 0) >= 0.3) {
//         row.classList.add('overlap'); num.textContent = '⚠';
//         num.title = `Có người nói chồng ~${Math.round(r.overlap * 100)}% câu này: người nói/cảm xúc kém tin cậy, hãy kiểm tra.`;
//       }
//       if (r.overlap_added) {
//         row.classList.add('added');
//         num.title = 'Dòng thêm từ giọng đã tách nguồn (người nói chồng/chen ngang). Chữ có thể sai: bấm phát để nghe và sửa.';
//       }
//       const spk = mkInput(r.speaker, 'text', v => { r.speaker = v; });
//       spk.setAttribute('list', 'speakerList');
//       spk.addEventListener('change', renderSlots);
//       const st = mkInput(r.start, 'number', v => { r.start = parseFloat(v); }, {step: 0.01, min: 0});
//       const en = mkInput(r.end, 'number', v => { r.end = parseFloat(v); }, {step: 0.01, min: 0});
//       const tx = mkInput(r.text, 'text', v => { r.text = v; }, {maxLength: 1500});
//       const res = document.createElement('div'); res.className = 'res';
//       const hit = byIdx.get(i);
//       if (hit) {
//         const chip = document.createElement('div'); chip.className = 'emotion-chip';
//         chip.textContent = `${faces[hit.emotion] || '•'} ${names[hit.emotion] || hit.emotion}`;
//         const small = document.createElement('small');
//         const top3 = [...hit.probabilities].sort((a, b) => b.percent - a.percent).slice(0, 3)
//           .map(p => `${names[p.label] || p.label} ${p.percent.toFixed(0)}%`).join(' · ');
//         small.textContent = `${hit.confidence.toFixed(1)}% · ${top3}` + (hit.truth ? ` · đúng: ${names[hit.truth] || hit.truth}` : '');
//         res.append(chip, small);
//         if (state.meld) {
//           res.classList.add('model');
//           small.textContent = `${hit.confidence.toFixed(1)}%`;
//           if (hit.correct !== null && hit.correct !== undefined) {
//             const v = document.createElement('div'); v.className = 'verdict';
//             v.textContent = hit.correct ? '✓ ĐÚNG' : '✗ SAI';
//             res.append(v);
//           }
//         }
//         if (hit.correct === true) row.classList.add('right');
//         if (hit.correct === false) row.classList.add('wrong');
//       }
//       const csvRes = document.createElement('div'); csvRes.className = 'res csvres';
//       if (state.meld) {
//         const c = document.createElement('div'); c.className = 'emotion-chip';
//         c.textContent = r.truth ? `${faces[r.truth] || '•'} ${names[r.truth] || r.truth}` : '—';
//         csvRes.append(c);
//       }
//       const play = document.createElement('button'); play.type = 'button'; play.className = 'mini';
//       play.innerHTML = '<svg class="icon"><use href="#i-play"/></svg>';
//       play.setAttribute('aria-label', 'Phát đoạn này');
//       play.addEventListener('click', () => playRow(i));
//       const del = document.createElement('button'); del.type = 'button'; del.className = 'mini';
//       del.innerHTML = '<svg class="icon"><use href="#i-trash"/></svg>';
//       del.setAttribute('aria-label', 'Xóa dòng');
//       del.addEventListener('click', () => { state.rows.splice(i, 1); state.results = null; renderAll(); });
//       if (state.meld) row.append(num, spk, st, en, tx, res, csvRes, play, del);
//       else row.append(num, spk, st, en, tx, res, play, del);
//       return row;
//     });
//     table.replaceChildren(head, ...rows);
//     $('rowCount').textContent = state.rows.length;
//   }

//   function renderSummary() {
//     const box = $('summary');
//     const r = state.results;
//     $('dlResultBtn').classList.toggle('hidden', !r);
//     if (!r) { box.classList.add('hidden'); box.replaceChildren(); return; }
//     const pills = Object.entries(r.counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => {
//       const p = document.createElement('span'); p.className = 'pill'; p.textContent = `${faces[k] || ''} ${names[k] || k}: ${v}`; return p;
//     });
//     if (r.accuracy !== null && r.accuracy !== undefined) {
//       const a = document.createElement('span'); a.className = 'pill good';
//       a.textContent = `Độ chính xác so với cột Emotion: ${r.accuracy}% (${r.graded} câu có nhãn)`; pills.unshift(a);
//     }
//     box.replaceChildren(...pills); box.classList.remove('hidden');
//   }

//   function renderAll() { renderSlots(); renderTable(); renderSummary(); renderReport(); }

//   $('addRowBtn').addEventListener('click', () => {
//     const last = state.rows[state.rows.length - 1];
//     const s = last ? last.end : 0;
//     state.rows.push({start: s, end: s + 2, speaker: last ? last.speaker : 'Speaker 1', text: '', dialogue_id: null, truth: null});
//     state.results = null; renderAll();
//   });

//   function playRow(i) {
//     const r = state.rows[i], v = $('player');
//     if (!v.src) return;
//     v.currentTime = r.start; state.stopAt = r.end; v.play().catch(() => {});
//     document.querySelectorAll('.utt-row').forEach(el => el.classList.toggle('playing', +el.dataset.i === i));
//   }
//   $('player').addEventListener('timeupdate', () => {
//     const v = $('player');
//     if (state.stopAt !== null && v.currentTime >= state.stopAt) { v.pause(); state.stopAt = null; document.querySelectorAll('.utt-row.playing').forEach(el => el.classList.remove('playing')); }
//   });
//   $('player').addEventListener('seeking', () => { /* manual seek cancels the auto-stop only when far from the row */ });

//   /* ---------- CSV ---------- */
//   const csvCell = (v) => { const s = String(v ?? ''); return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
//   function download(name, text) {
//     const blob = new Blob(['\ufeff' + text], {type: 'text/csv;charset=utf-8'});
//     const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
//     document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
//   }
//   $('dlInputBtn').addEventListener('click', () => {
//     const rows = [...state.rows].sort((a, b) => a.start - b.start);
//     const lines = [['Dialogue_ID', 'Utterance_ID', 'Speaker', 'Utterance', 'StartTime', 'EndTime', 'Emotion'].join(',')];
//     rows.forEach((r, i) => lines.push([r.dialogue_id ?? 0, i, r.speaker, r.text, r.start, r.end, r.truth ?? ''].map(csvCell).join(',')));
//     download('utterances_input.csv', lines.join('\r\n'));
//   });
//   $('dlResultBtn').addEventListener('click', () => { if (state.resultCsv) download('digemo_predictions.csv', state.resultCsv); });

//   /* ---------- predict ---------- */
//   async function predict() {
//     if (state.busy || !state.token) return;
//     const bad = state.rows.findIndex(r => !(r.text || '').trim() || !(r.speaker || '').trim() || !(r.end > r.start));
//     if (bad >= 0) { toast(`Dòng ${bad + 1}: cần speaker, transcript và thời gian kết thúc lớn hơn bắt đầu.`); return; }
//     state.rows.sort((a, b) => a.start - b.start);
//     renderAll();
//     const body = {
//       token: state.token,
//       rows: state.rows.map(r => ({start: r.start, end: r.end, speaker: r.speaker.trim(), text: r.text.trim(), dialogue_id: r.dialogue_id ?? null, truth: r.truth ?? null})),
//       slots: state.slots,
//       max_per_dialogue: state.meld ? 100 : (Math.round(+$('maxPer').value) || 30),
//     };
//     setBusy(true);
//     try {
//       showProgress(0.01, 'Đang gửi yêu cầu…');
//       const start = await api('/api/video/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
//       const res = await pollJob(start.job_id);
//       state.results = res; state.resultCsv = res.csv; state.slots = {...state.slots, ...res.slots};
//       if (state.meld) recordRun(res);
//       renderAll();
//       setBusy(false, 'XONG');
//       if (window.updateStepper) window.updateStepper(3);
//     } catch (e) { toast(e.message); setBusy(false, 'LỖI'); }
//   }
//   $('predictBtn').addEventListener('click', predict);

//   /* ---------- MELD mode: toggle + preview ---------- */
//   const EMO = Object.keys(names);
//   const short = (l) => (names[l] || l).slice(0, 3).toUpperCase();
//   const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };

//   function syncMeldUi() {
//     const on = $('meldMode').checked;
//     $('meldBox').classList.toggle('hidden', !on);
//     $('csvDrop').classList.toggle('hidden', on);
//     $('speakerOpt').classList.toggle('hidden', on);
//     $('autoHint').classList.toggle('hidden', on);
//   }
//   $('meldMode').addEventListener('change', syncMeldUi);

//   async function previewMeld() {
//     const id = parseInt($('meldId').value, 10);
//     if (!Number.isInteger(id) || id < 0) { toast('Hãy nhập Dialogue_ID hợp lệ (số nguyên ≥ 0).'); return; }
//     const box = $('meldPreview');
//     try {
//       const d = await api(`/api/video/meld/${id}?time_mode=${encodeURIComponent($('meldTime').value)}`);
//       const counts = {};
//       d.rows.forEach(r => { if (r.truth) counts[r.truth] = (counts[r.truth] || 0) + 1; });
//       const pills = [`Dialogue ${d.dialogue_id}`, `${d.count} câu`, `${d.duration}s`, `Người nói: ${d.speakers.join(', ')}`].map(t => el('span', 'pill', t));
//       Object.entries(counts).forEach(([k, v]) => pills.push(el('span', 'pill', `${faces[k] || ''} ${names[k] || k}: ${v}`)));
//       box.replaceChildren(...pills); box.classList.remove('hidden');
//     } catch (e) { box.classList.add('hidden'); toast(e.message); }
//   }
//   $('meldLoadBtn').addEventListener('click', previewMeld);

//   /* ---------- run history (localStorage, latest run per dialogue) ---------- */
//   const RUNS_KEY = 'digemo_meld_runs';
//   const loadRuns = () => { try { return JSON.parse(localStorage.getItem(RUNS_KEY) || '[]'); } catch (_) { return []; } };
//   const saveRuns = (runs) => { try { localStorage.setItem(RUNS_KEY, JSON.stringify(runs.slice(-40))); } catch (_) {} };
//   function recordRun(res) {
//     if (res.graded <= 0) return;
//     const correct = res.results.filter(r => r.correct === true).length;
//     const runs = loadRuns().filter(r => r.dia !== state.meldDia);
//     runs.push({dia: state.meldDia, correct, graded: res.graded, ts: Date.now()});
//     saveRuns(runs);
//   }
//   $('clearRunsBtn').addEventListener('click', () => { saveRuns([]); renderReport(); });

//   /* ---------- report charts ---------- */
//   function renderReport() {
//     const card = $('reportCard');
//     const res = state.results;
//     if (!state.meld || !res) { card.classList.add('hidden'); return; }
//     const items = res.results;
//     const graded = items.filter(r => r.truth);
//     const ok = graded.filter(r => r.correct === true).length;
//     const bad = graded.length - ok;
//     const acc = graded.length ? ok / graded.length : 0;
//     card.classList.remove('hidden');
//     $('reportDia').textContent = `#${state.meldDia}`;
//     $('reportTag').textContent = graded.length ? `${ok}/${graded.length} CÂU ĐÚNG` : 'KHÔNG CÓ NHÃN';

//     // donut: red base ring, green arc on top
//     const C = 2 * Math.PI * 48;
//     const ns = 'http://www.w3.org/2000/svg';
//     const ring = (stroke, dash) => {
//       const c = document.createElementNS(ns, 'circle');
//       c.setAttribute('cx', 60); c.setAttribute('cy', 60); c.setAttribute('r', 48);
//       c.setAttribute('fill', 'none'); c.setAttribute('stroke', stroke); c.setAttribute('stroke-width', 14);
//       if (dash !== undefined) { c.setAttribute('stroke-dasharray', `${dash} ${C}`); c.setAttribute('stroke-linecap', 'round'); }
//       c.setAttribute('transform', 'rotate(-90 60 60)');
//       return c;
//     };
//     $('donut').replaceChildren(
//       ring(graded.length ? '#ff6f86' : '#28374b'),
//       ...(graded.length && ok > 0 ? [ring('#5be4b7', Math.max(0.1, acc * C - (ok < graded.length ? 3 : 0)))] : []));
//     $('donutPct').textContent = graded.length ? `${(acc * 100).toFixed(0)}%` : '—';
//     $('reportStats').replaceChildren(
//       ...[['Câu có nhãn', graded.length, ''], ['Dự đoán đúng', ok, 'good'], ['Dự đoán sai', bad, 'bad']].map(([t, v, c]) => {
//         const s = el('div', `stat ${c}`); s.append(el('small', '', t), el('strong', '', String(v))); return s;
//       }));

//     // timeline
//     $('timeline').replaceChildren(...items.map((r, i) => {
//       const cls = !r.truth ? 'na' : r.correct ? 'ok' : 'no';
//       const c = el('div', `tl-cell ${cls}`);
//       c.append(el('span', '', !r.truth ? '·' : r.correct ? '✓' : '✗'), document.createTextNode(String(i + 1)));
//       c.title = `#${i + 1} ${r.speaker}: model = ${names[r.emotion] || r.emotion}` + (r.truth ? ` · CSV = ${names[r.truth] || r.truth}` : '');
//       return c;
//     }));

//     // per-emotion stacked bars (by true label)
//     const per = Object.fromEntries(EMO.map(l => [l, {ok: 0, no: 0}]));
//     graded.forEach(r => { if (per[r.truth]) per[r.truth][r.correct ? 'ok' : 'no']++; });
//     $('emoBars').replaceChildren(...EMO.map(l => {
//       const {ok: a, no: b} = per[l], n = a + b;
//       const row = el('div', 'emo-row'); row.append(el('span', 'nm', `${faces[l]} ${names[l]}`));
//       const track = el('div', 'emo-track');
//       if (n) {
//         const g = el('i', 'g'); g.style.width = `${a / n * 100}%`;
//         const r = el('i', 'r'); r.style.width = `${b / n * 100}%`;
//         track.append(g, r);
//       }
//       row.append(track, el('span', 'ct', n ? `${a}/${n}` : '—'));
//       return row;
//     }));

//     // confusion matrix (rows = CSV truth, cols = model)
//     const idx = Object.fromEntries(EMO.map((l, i) => [l, i]));
//     const M = EMO.map(() => EMO.map(() => 0));
//     graded.forEach(r => { if (r.truth in idx && r.emotion in idx) M[idx[r.truth]][idx[r.emotion]]++; });
//     const cells = [el('span', 'h', 'THẬT↓')];
//     EMO.forEach(l => cells.push(el('span', 'h', short(l))));
//     M.forEach((rowv, i) => {
//       const tot = rowv.reduce((a, b) => a + b, 0);
//       cells.push(el('span', 'h', short(EMO[i])));
//       rowv.forEach((v, j) => {
//         const c = el('span', `c${i === j ? ' diag' : ''}`, v ? String(v) : '');
//         if (v) {
//           const f = v / tot;
//           c.style.background = i === j ? `rgba(101,230,216,${0.25 + 0.75 * f})` : `rgba(255,111,134,${0.25 + 0.75 * f})`;
//           if (f > 0.5) c.classList.add('hit');
//         }
//         c.title = `Thật: ${names[EMO[i]]} → Model: ${names[EMO[j]]} (${v})`;
//         cells.push(c);
//       });
//     });
//     const conf = el('div', 'conf'); conf.append(...cells);
//     $('confMatrix').replaceChildren(conf);

//     // run history
//     const runs = loadRuns();
//     const list = $('runList');
//     if (!runs.length) { list.replaceChildren(el('div', 'run-empty', 'Chưa có dialogue nào được ghi lại.')); $('runTotal').textContent = ''; }
//     else {
//       list.replaceChildren(...runs.map(r => {
//         const row = el('div', 'run-item');
//         const bar = el('div', 'bar'), fill = el('i'); fill.style.width = `${r.correct / r.graded * 100}%`; bar.append(fill);
//         row.append(el('span', '', `Dialogue ${r.dia}`), bar, el('span', 'meta', `${r.correct}/${r.graded} · ${(r.correct / r.graded * 100).toFixed(0)}%`));
//         return row;
//       }));
//       const C2 = runs.reduce((a, r) => a + r.correct, 0), G2 = runs.reduce((a, r) => a + r.graded, 0);
//       $('runTotal').textContent = `TỔNG ${runs.length} DIALOGUE: ${C2}/${G2} = ${(C2 / G2 * 100).toFixed(1)}%`;
//     }
//   }

//   window.addEventListener('beforeunload', () => { if (state.token) navigator.sendBeacon?.(`/api/video/${state.token}/delete`); });
//   setBusy(false, 'CHƯA CHẠY');
// })();

/* DiGemo video analysis page: upload -> editable utterance table -> prediction -> analytics.
 *
 * Data model (single source of truth for table, analytics and CSV):
 *   state.rows   : ordered utterances. Each has a client-side stable `uid`
 *                  (never sent to the API), plus text_original (from ASR/CSV) and text (current).
 *   state.preds  : Map(uid -> { res, snap })
 *                  res  = one item of the /predict response, bound to the row by src_index
 *                         at the moment of the call.
 *                  snap = what the model actually saw (speaker, start, end, text, slot).
 *   Everything that is displayed or exported is derived from current(), which joins rows and
 *   preds by uid. Nothing is looked up by array position after the response has been bound.
 */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  // Label names and class order MUST match the model checkpoint. Do not reorder.
  const names = { neutral: 'Neutral', surprise: 'Surprise', fear: 'Fear', sadness: 'Sadness', joy: 'Joy', disgust: 'Disgust', anger: 'Anger' };
  const EMO = Object.keys(names);
  const short = (l) => (names[l] || l).slice(0, 3).toUpperCase();
  const emoCls = (l) => (EMO.includes(l) ? `emo-${l}` : '');
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };

  const state = {
    token: null, rows: [], slots: {}, nSlots: 9,
    preds: new Map(), results: null, resultCsv: '',
    busy: false, videoUrl: null, stopAt: null, meld: false, meldDia: null,
    edit: true, seq: 0, filter: { speaker: '', emotion: '', q: '' }, tlSpeaker: ''
  };
  const newUid = () => `u${++state.seq}`;
  const adopt = (r) => ({ ...r, uid: newUid(), text_original: r.text_original ?? r.text ?? '' });

  /* ---------- generic helpers ---------- */
  let toastTimer = null;
  const toast = (msg, kind = 'error') => {
    $('toastText').textContent = msg;
    $('toast').className = `toast ${kind === 'error' ? '' : kind}`.trim();
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $('toast').classList.add('hidden'), kind === 'error' ? 9000 : 6000);
  };
  $('toastClose').addEventListener('click', () => $('toast').classList.add('hidden'));
  const api = async (path, options = {}) => {
    const r = await fetch(path, options);
    let data;
    try { data = await r.json(); } catch (_) { throw new Error(`HTTP ${r.status}: máy chủ không phản hồi JSON.`); }
    if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${r.status}`);
    return data;
  };
  const fmt = (s) => { s = Math.max(0, +s || 0); const m = Math.floor(s / 60); return `${m}:${(s - m * 60).toFixed(2).padStart(5, '0')}`; };
  const fmtSize = (b) => b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`;
  const pad2 = (n) => String(n).padStart(2, '0');

  function setBusy(flag, tag) {
    state.busy = flag;
    $('analyzeBtn').disabled = flag || !$('videoFile').files[0];
    $('clearBtn').disabled = flag || !state.token;
    $('predictBtn').disabled = flag;
    $('addRowBtn').disabled = flag;
    const label = tag || (flag ? 'ĐANG XỬ LÝ' : 'SẴN SÀNG');
    $('jobTag').textContent = label;
    $('jobTag').className = 'tag ' + (flag ? 'busy' : (label === 'LỖI' ? 'recording' : 'ready'));
    const pill = $('appStatus');
    pill.classList.toggle('busy', flag);
    pill.classList.toggle('error', !flag && label === 'LỖI');
    $('appStatusText').textContent = flag ? 'Đang xử lý' : (label === 'LỖI' ? 'Có lỗi' : 'Sẵn sàng');
    $('skeleton').classList.toggle('hidden', !flag);
  }
  function showProgress(p, msg) {
    $('progress').classList.remove('hidden');
    $('progressBar').style.width = `${Math.round(p * 100)}%`;
    $('progressText').textContent = `${Math.round(p * 100)}% · ${msg || ''}`;
  }
  async function pollJob(id) {
    for (; ;) {
      const j = await api(`/api/video/jobs/${id}`);
      showProgress(j.progress, j.message);
      if (j.status === 'done') return j.result;
      if (j.status === 'error') throw new Error(j.error || 'Tác vụ thất bại.');
      await new Promise(r => setTimeout(r, 900));
    }
  }
  function syncStep() {
    const n = !state.token ? 1 : (state.preds.size ? 3 : 2);
    if (window.updateStepper) window.updateStepper(n);
  }

  /* ---------- canonical join of rows + predictions ---------- */
  function isStale(row, p) {
    const s = p.snap;
    return (row.speaker || '').trim() !== s.speaker
      || Number(row.start) !== s.start || Number(row.end) !== s.end
      || (row.text || '').trim() !== s.text
      || state.slots[s.speaker] !== s.slot;
  }
  function current() {
    return state.rows.map((row, i) => {
      const pred = state.preds.get(row.uid) || null;
      return { row, i, pred, stale: !!pred && isStale(row, pred) };
    });
  }
  const speakerNames = () => [...new Set(state.rows.map(r => (r.speaker || '').trim()).filter(Boolean))];

  /* ---------- step 1: files ---------- */
  const VIDEO_EXT = ['mp4', 'avi', 'mov', 'mkv', 'webm'];
  const MAX_BYTES = 800 * 1024 * 1024;
  function setVideoStatus(text, cls) {
    const c = $('videoMeta').querySelector('.status');
    if (c) { c.textContent = text; c.className = `meta-chip status ${cls || ''}`.trim(); }
  }
  function renderVideoMeta(f) {
    const box = $('videoMeta');
    if (!f) { box.classList.add('hidden'); box.replaceChildren(); return; }
    const ext = (f.name.split('.').pop() || '').toUpperCase();
    box.replaceChildren(el('span', 'meta-chip', fmtSize(f.size)), el('span', 'meta-chip', ext || 'VIDEO'), el('span', 'meta-chip status', 'Sẵn sàng tải lên'));
    box.classList.remove('hidden');
  }
  function pickCsv() {
    const f = $('csvFile').files[0];
    $('csvName').textContent = f ? f.name : 'CSV utterance (tùy chọn)';
    $('csvDrop').classList.toggle('has-file', !!f);
  }
  $('videoFile').addEventListener('change', () => {
    const f = $('videoFile').files[0];
    if (f) {
      const ext = (f.name.split('.').pop() || '').toLowerCase();
      if (!VIDEO_EXT.includes(ext)) { toast('Định dạng chưa hỗ trợ. Hãy chọn MP4, AVI, MOV, MKV hoặc WebM.'); $('videoFile').value = ''; }
      else if (f.size > MAX_BYTES) { toast('Video lớn hơn 800 MB.'); $('videoFile').value = ''; }
    }
    const ok = $('videoFile').files[0];
    $('videoName').textContent = ok ? ok.name : 'Kéo video vào đây hoặc bấm để chọn';
    $('videoDrop').classList.toggle('has-file', !!ok);
    renderVideoMeta(ok);
    $('analyzeBtn').disabled = state.busy || !ok;
  });
  $('csvFile').addEventListener('change', () => {
    pickCsv();
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
    setVideoStatus('Đang tải lên & phân đoạn', 'busy');
    try {
      showProgress(0.01, 'Đang tải video lên…');
      const start = await api('/api/video/analyze', { method: 'POST', body: form });
      state.token = start.token;
      const res = await pollJob(start.job_id);
      state.rows = res.rows.map(adopt);
      state.preds = new Map();
      state.slots = res.slots;
      state.nSlots = res.n_slots || 9;
      state.results = null; state.resultCsv = '';
      state.meld = res.source === 'meld';
      state.meldDia = state.meld ? res.meld_dialogue : null;
      if (res.warning) toast(res.warning, 'warn');
      if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
      state.videoUrl = URL.createObjectURL(file);
      $('player').src = state.videoUrl;
      $('reviewGrid').classList.remove('hidden');
      $('tableCard').classList.remove('hidden');
      renderAll();
      setBusy(false, 'ĐÃ PHÂN ĐOẠN');
      setVideoStatus('Đã phân đoạn', 'done');
      syncStep();
    } catch (e) {
      toast(e.message);
      if (state.token) { fetch(`/api/video/${state.token}`, { method: 'DELETE' }).catch(() => { }); state.token = null; }
      setBusy(false, 'LỖI');
      setVideoStatus('Lỗi xử lý', '');
      syncStep();
    }
  }
  async function clearVideo(silent) {
    if (state.token) await fetch(`/api/video/${state.token}`, { method: 'DELETE' }).catch(() => { });
    state.token = null; state.rows = []; state.preds = new Map(); state.results = null; state.resultCsv = '';
    state.meld = false; state.meldDia = null;
    ['reportCard', 'reviewGrid', 'tableCard', 'analyticsCard', 'progress'].forEach(id => $(id).classList.add('hidden'));
    if (!silent) {
      setBusy(false, 'CHƯA CHẠY');
      setVideoStatus('Sẵn sàng tải lên', '');
      syncStep();
    }
  }
  $('analyzeBtn').addEventListener('click', analyze);
  $('clearBtn').addEventListener('click', () => clearVideo(false));

  /* ---------- step 2: slots, filters ---------- */
  function renderSlots() {
    const used = new Set();
    const items = speakerNames().map(s => {
      let v = state.slots[s];
      if (v === undefined || v === null) { v = [...Array(state.nSlots).keys()].find(i => !used.has(i)) ?? state.nSlots - 1; state.slots[s] = v; }
      used.add(v);
      const row = el('div', 'slot-item');
      const sel = document.createElement('select');
      sel.setAttribute('aria-label', `Slot cho ${s}`);
      for (let i = 0; i < state.nSlots; i++) sel.add(new Option(i === state.nSlots - 1 ? `Slot ${i} (Other)` : `Slot ${i}`, i));
      sel.value = v;
      sel.addEventListener('change', () => { state.slots[s] = +sel.value; refreshStaleViews(); });
      row.append(el('span', '', s), sel);
      return row;
    });
    $('slotList').replaceChildren(...items);
    $('speakerList').replaceChildren(...speakerNames().map(s => new Option(s)));
  }

  function syncFilterOptions() {
    const fill = (sel, first, opts, key) => {
      const keep = state.filter[key];
      sel.replaceChildren(new Option(first, ''), ...opts.map(([v, t]) => new Option(t, v)));
      sel.value = opts.some(([v]) => v === keep) ? keep : '';
      state.filter[key] = sel.value;
    };
    const spk = speakerNames().map(s => [s, s]);
    fill($('fSpeaker'), 'Mọi speaker', spk, 'speaker');
    fill($('fEmotion'), 'Mọi cảm xúc', EMO.map(l => [l, names[l]]), 'emotion');
    const tl = $('tlSpeaker'), keep = state.tlSpeaker;
    tl.replaceChildren(new Option('Mọi speaker', ''), ...spk.map(([v, t]) => new Option(t, v)));
    tl.value = spk.some(([v]) => v === keep) ? keep : ''; state.tlSpeaker = tl.value;
  }
  function passes(c) {
    const f = state.filter;
    if (f.speaker && (c.row.speaker || '').trim() !== f.speaker) return false;
    if (f.emotion && !(c.pred && c.pred.res.emotion === f.emotion)) return false;
    if (f.q) {
      const hay = `${c.row.text || ''} ${c.row.speaker || ''}`.toLowerCase();
      if (!hay.includes(f.q.toLowerCase())) return false;
    }
    return true;
  }
  $('fSpeaker').addEventListener('change', () => { state.filter.speaker = $('fSpeaker').value; renderTable(); });
  $('fEmotion').addEventListener('change', () => { state.filter.emotion = $('fEmotion').value; renderTable(); });
  let qTimer = null;
  $('fQuery').addEventListener('input', () => { clearTimeout(qTimer); qTimer = setTimeout(() => { state.filter.q = $('fQuery').value.trim(); renderTable(); }, 150); });
  $('tlSpeaker').addEventListener('change', () => { state.tlSpeaker = $('tlSpeaker').value; renderTimeline(); });
  $('editToggle').addEventListener('click', () => {
    state.edit = !state.edit;
    $('editToggle').classList.toggle('active', state.edit);
    $('editToggle').setAttribute('aria-pressed', String(state.edit));
    renderTable();
  });

  /* ---------- step 2: table ---------- */
  function buildRes(row, pred, stale) {
    const res = el('div', 'res');
    if (!pred) { res.append(el('span', 'empty-res', '—')); return res; }
    const it = pred.res;
    const conf = Number(it.confidence);
    res.append(el('div', `emotion-chip ${emoCls(it.emotion)}`, names[it.emotion] || it.emotion));
    res.append(el('span', 'conf', Number.isFinite(conf) ? `${conf.toFixed(1)}%` : '—'));
    if (state.meld) {
      res.classList.add('model');
      if (!stale && it.correct !== null && it.correct !== undefined) res.append(el('div', 'verdict', it.correct ? '✓ ĐÚNG' : '✗ SAI'));
    } else {
      const top3 = [...(it.probabilities || [])].sort((a, b) => b.percent - a.percent).slice(0, 3)
        .map(p => `${names[p.label] || p.label} ${p.percent.toFixed(0)}%`).join(' · ');
      const truth = row.truth || it.truth;
      res.append(el('small', '', top3 + (truth ? ` · đúng: ${names[truth] || truth}` : '')));
    }
    if (stale) { res.classList.add('is-stale'); res.append(el('span', 'stale-badge', 'Đã sửa · dự đoán cũ')); }
    return res;
  }
  function applyRowState(tr, pred, stale) {
    tr.classList.remove('right', 'wrong');
    if (pred && !stale) {
      if (pred.res.correct === true) tr.classList.add('right');
      if (pred.res.correct === false) tr.classList.add('wrong');
    }
  }
  function refreshRow(tr, row) {
    const pred = state.preds.get(row.uid) || null;
    const stale = !!pred && isStale(row, pred);
    tr.querySelector('.res').replaceWith(buildRes(row, pred, stale));
    applyRowState(tr, pred, stale);
    renderStaleNote();
  }
  const autosize = (ta) => { ta.style.height = 'auto'; ta.style.height = `${Math.min(ta.scrollHeight + 2, 150)}px`; };

  function buildRow(c) {
    const { row, i, pred, stale } = c;
    const tr = el('div', 'utt-row'); tr.dataset.uid = row.uid;
    const num = el('span', 'num', pad2(i + 1));
    if ((row.overlap || 0) >= 0.3) {
      tr.classList.add('overlap'); num.textContent = '⚠';
      num.title = `Có người nói chồng ~${Math.round(row.overlap * 100)}% câu này: người nói/cảm xúc kém tin cậy, hãy kiểm tra.`;
    }
    if (row.overlap_added) {
      tr.classList.add('added');
      num.title = 'Dòng thêm từ giọng đã tách nguồn (người nói chồng/chen ngang). Chữ có thể sai: bấm phát để nghe và sửa.';
    }
    const after = () => refreshRow(tr, row);
    let spk, st, en, tx;
    if (state.edit) {
      spk = el('input'); spk.type = 'text'; spk.value = row.speaker || ''; spk.setAttribute('list', 'speakerList'); spk.setAttribute('aria-label', `Speaker dòng ${i + 1}`);
      spk.addEventListener('input', () => { row.speaker = spk.value; after(); });
      spk.addEventListener('change', () => { renderSlots(); syncFilterOptions(); refreshStaleViews(); });
      st = el('input'); st.type = 'number'; st.step = '0.01'; st.min = '0'; st.value = row.start; st.setAttribute('aria-label', `Bắt đầu dòng ${i + 1}`);
      st.addEventListener('input', () => { row.start = parseFloat(st.value); after(); });
      en = el('input'); en.type = 'number'; en.step = '0.01'; en.min = '0'; en.value = row.end; en.setAttribute('aria-label', `Kết thúc dòng ${i + 1}`);
      en.addEventListener('input', () => { row.end = parseFloat(en.value); after(); });
      tx = el('textarea'); tx.rows = 1; tx.maxLength = 1500; tx.value = row.text || ''; tx.setAttribute('aria-label', `Transcript dòng ${i + 1}`);
      tx.addEventListener('input', () => { row.text = tx.value; autosize(tx); after(); });
      [st, en, tx].forEach(x => x.addEventListener('change', refreshStaleViews));
    } else {
      spk = el('div', 'cell-text speaker', row.speaker || '—');
      st = el('div', 'cell-text time', Number(row.start).toFixed(2));
      en = el('div', 'cell-text time', Number(row.end).toFixed(2));
      tx = el('div', 'cell-text transcript', row.text || '—');
    }
    const res = buildRes(row, pred, stale);
    const play = el('button', 'mini'); play.type = 'button'; play.setAttribute('aria-label', 'Phát đoạn này');
    play.innerHTML = '<svg class="icon"><use href="#i-play"/></svg>';
    play.addEventListener('click', () => playRow(row.uid));
    const del = el('button', 'mini del'); del.type = 'button'; del.setAttribute('aria-label', 'Xóa dòng');
    del.innerHTML = '<svg class="icon"><use href="#i-trash"/></svg>';
    del.addEventListener('click', () => {
      state.rows = state.rows.filter(r => r.uid !== row.uid);
      state.preds.delete(row.uid);
      renderAll(); syncStep();
    });
    if (state.meld) {
      const csvRes = el('div', 'res csvres');
      csvRes.append(row.truth ? el('div', `emotion-chip ${emoCls(row.truth)}`, names[row.truth] || row.truth) : el('span', 'empty-res', '—'));
      tr.append(num, spk, st, en, tx, res, csvRes, play, del);
    } else tr.append(num, spk, st, en, tx, res, play, del);
    applyRowState(tr, pred, stale);
    return tr;
  }

  function renderTable() {
    const table = $('uttTable');
    table.classList.toggle('meld', state.meld);
    const head = el('div', 'utt-row head');
    const heads = state.meld
      ? ['#', 'Speaker', 'Bắt đầu (s)', 'Kết thúc (s)', 'Transcript', 'Model dự đoán', 'Nhãn CSV', '', '']
      : ['#', 'Speaker', 'Bắt đầu (s)', 'Kết thúc (s)', 'Transcript', 'Cảm xúc · độ tin cậy', '', ''];
    heads.forEach(t => head.append(el('span', '', t)));
    const all = current();
    const vis = all.filter(passes);
    const rows = vis.map(buildRow);
    table.replaceChildren(head, ...(rows.length ? rows : [el('div', 'table-empty', state.rows.length ? 'Không có dòng nào khớp bộ lọc.' : 'Chưa có utterance.')]));
    $('rowCount').textContent = state.rows.length;
    $('filterInfo').textContent = vis.length !== all.length ? `Hiển thị ${vis.length}/${all.length}` : '';
    if (state.edit) requestAnimationFrame(() => table.querySelectorAll('textarea').forEach(autosize));
  }

  function renderSummary() {
    const box = $('summary');
    const cur = current().filter(c => c.pred);
    $('dlResultBtn').classList.toggle('hidden', !cur.length);
    $('dlServerBtn').classList.toggle('hidden', !state.resultCsv);
    if (!cur.length) { box.classList.add('hidden'); box.replaceChildren(); return; }
    const counts = {};
    cur.forEach(c => { counts[c.pred.res.emotion] = (counts[c.pred.res.emotion] || 0) + 1; });
    const pills = EMO.filter(k => counts[k]).sort((a, b) => counts[b] - counts[a])
      .map(k => el('span', `pill ${emoCls(k)}`, `${names[k]}: ${counts[k]}`));
    const graded = cur.filter(c => !c.stale && (c.row.truth || c.pred.res.truth) && typeof c.pred.res.correct === 'boolean');
    if (graded.length) {
      const ok = graded.filter(c => c.pred.res.correct).length;
      pills.unshift(el('span', 'pill good', `Độ chính xác so với cột Emotion: ${(ok / graded.length * 100).toFixed(1)}% (${graded.length} câu có nhãn)`));
    }
    box.replaceChildren(...pills); box.classList.remove('hidden');
  }
  function renderStaleNote() {
    const n = current().filter(c => c.stale).length;
    const box = $('staleNote');
    box.classList.toggle('hidden', !n);
    if (n) box.textContent = `${n} dòng đã được sửa (transcript, speaker, thời gian hoặc slot) sau khi dự đoán. Kết quả cũ được giữ nguyên và đánh dấu, không tự đổi theo nội dung mới; bấm "Dự đoán cảm xúc" để cập nhật.`;
  }
  function refreshStaleViews() { renderTable(); renderSummary(); renderStaleNote(); renderAnalytics(); renderReport(); }
  function renderAll() { syncFilterOptions(); renderSlots(); renderTable(); renderSummary(); renderStaleNote(); renderAnalytics(); renderReport(); }

  $('addRowBtn').addEventListener('click', () => {
    const last = state.rows[state.rows.length - 1];
    const s = last ? Number(last.end) || 0 : 0;
    const row = adopt({ start: s, end: s + 2, speaker: last ? last.speaker : 'Speaker 1', text: '', text_original: '', dialogue_id: last ? last.dialogue_id ?? null : null, truth: null });
    state.rows.push(row);
    renderAll();
    const tr = document.querySelector(`.utt-row[data-uid="${row.uid}"]`);
    if (tr) { tr.classList.add('flash'); tr.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
  });

  function playRow(uid) {
    const r = state.rows.find(x => x.uid === uid), v = $('player');
    if (!r || !v.src) return;
    v.currentTime = r.start; state.stopAt = r.end; v.play().catch(() => { });
    document.querySelectorAll('.utt-row').forEach(e => e.classList.toggle('playing', e.dataset.uid === uid));
  }
  $('player').addEventListener('timeupdate', () => {
    const v = $('player');
    if (state.stopAt !== null && v.currentTime >= state.stopAt) {
      v.pause(); state.stopAt = null;
      document.querySelectorAll('.utt-row.playing').forEach(e => e.classList.remove('playing'));
    }
  });
  $('player').addEventListener('play', () => document.body.classList.add('video-playing'));
  ['pause', 'ended'].forEach(ev => $('player').addEventListener(ev, () => document.body.classList.remove('video-playing')));

  /* ---------- CSV (both exports are built from current(), i.e. exactly what the table shows) ---------- */
  const csvCell = (v) => { const s = String(v ?? ''); return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  function download(name, text) {
    const body = text.charCodeAt(0) === 0xFEFF ? text : '\ufeff' + text; // single UTF-8 BOM so Excel reads Vietnamese correctly
    const blob = new Blob([body], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  function countCsvRecords(text) {
    let n = 0, inQ = false, any = false;
    for (let k = 0; k < text.length; k++) {
      const ch = text[k];
      if (ch === '"') { if (inQ && text[k + 1] === '"') k++; else inQ = !inQ; any = true; }
      else if ((ch === '\n' || ch === '\r') && !inQ) { if (ch === '\r' && text[k + 1] === '\n') k++; if (any) { n++; any = false; } }
      else any = true;
    }
    return any ? n + 1 : n;
  }
  const perDialogueCounter = () => { const m = new Map(); return (d) => { const k = d ?? 0; const v = m.get(k) ?? 0; m.set(k, v + 1); return v; }; };

  // Same header as the training CSV so the file can be re-imported. Order = table order.
  function buildInputCsv() {
    const next = perDialogueCounter();
    const lines = [['Dialogue_ID', 'Utterance_ID', 'Speaker', 'Utterance', 'StartTime', 'EndTime', 'Emotion'].join(',')];
    state.rows.forEach(r => lines.push([r.dialogue_id ?? 0, next(r.dialogue_id), r.speaker, r.text, r.start, r.end, r.truth ?? ''].map(csvCell).join(',')));
    return lines.join('\r\n');
  }
  const RESULT_HEAD = ['Order', 'Row_ID', 'Dialogue_ID', 'Utterance_ID', 'Speaker', 'StartTime', 'EndTime',
    'Utterance_Original', 'Utterance_Inferred', 'Utterance_Current', 'Edited',
    'Emotion_Truth', 'Emotion_Pred', 'Confidence_Pct', 'Correct', 'Status', ...EMO.map(l => `Prob_${l}`)];
  function buildResultCsv() {
    const next = perDialogueCounter();
    const lines = [RESULT_HEAD.join(',')];
    current().forEach(({ row, i, pred, stale }) => {
      const s = pred ? pred.snap : null, it = pred ? pred.res : null;
      const probs = {};
      if (it && Array.isArray(it.probabilities)) it.probabilities.forEach(p => { probs[p.label] = p.percent; });
      const edited = (row.text || '').trim() !== (row.text_original || '').trim();
      const conf = it ? Number(it.confidence) : NaN;
      lines.push([
        i + 1, row.uid, row.dialogue_id ?? 0, next(row.dialogue_id),
        s ? s.speaker : row.speaker, s ? s.start : row.start, s ? s.end : row.end,
        row.text_original, s ? s.text : '', row.text, edited ? 1 : 0,
        row.truth ?? (it && it.truth) ?? '', it ? it.emotion : '', Number.isFinite(conf) ? conf.toFixed(2) : '',
        it && typeof it.correct === 'boolean' ? (it.correct ? 1 : 0) : '',
        pred ? (stale ? 'stale' : 'ok') : 'no_prediction',
        ...EMO.map(l => (probs[l] !== undefined ? Number(probs[l]).toFixed(2) : ''))
      ].map(csvCell).join(','));
    });
    return lines.join('\r\n');
  }
  $('dlInputBtn').addEventListener('click', () => { download('utterances_input.csv', buildInputCsv()); toast(`Đã xuất ${state.rows.length} dòng (đúng thứ tự bảng).`, 'ok'); });
  $('dlResultBtn').addEventListener('click', () => {
    const stale = current().filter(c => c.stale).length;
    download('digemo_predictions.csv', buildResultCsv());
    toast(`Đã xuất ${state.rows.length} dòng.` + (stale ? ` ${stale} dòng đã sửa sau dự đoán (cột Status = stale).` : ''), stale ? 'warn' : 'ok');
  });
  $('dlServerBtn').addEventListener('click', () => { if (state.resultCsv) download('digemo_predictions_server.csv', state.resultCsv); });

  /* ---------- predict ---------- */
  async function predict() {
    if (state.busy || !state.token) return;
    const bad = state.rows.findIndex(r => !(r.text || '').trim() || !(r.speaker || '').trim() || !(r.end > r.start));
    if (bad >= 0) { toast(`Dòng ${bad + 1}: cần speaker, transcript và thời gian kết thúc lớn hơn bắt đầu.`); return; }
    state.rows.sort((a, b) => a.start - b.start); // stable sort; the table is re-rendered in the exact order that is sent
    renderAll();
    const sent = state.rows.map(r => ({
      uid: r.uid, speaker: r.speaker.trim(), text: r.text.trim(), start: Number(r.start), end: Number(r.end),
      dialogue_id: r.dialogue_id ?? null, truth: r.truth ?? null
    }));
    const body = {
      token: state.token,
      rows: sent.map(({ start, end, speaker, text, dialogue_id, truth }) => ({ start, end, speaker, text, dialogue_id, truth })), // same contract as before
      slots: state.slots,
      max_per_dialogue: state.meld ? 100 : (Math.round(+$('maxPer').value) || 30),
    };
    setBusy(true);
    try {
      showProgress(0.01, 'Đang gửi yêu cầu…');
      const start = await api('/api/video/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const res = await pollJob(start.job_id);
      state.slots = { ...state.slots, ...res.slots };

      // Bind every returned item to the row that was sent at that position (src_index), by uid.
      const items = Array.isArray(res.results) ? res.results : [];
      const preds = new Map();
      let spkMismatch = 0, unbound = 0;
      items.forEach((it, k) => {
        const si = Number.isInteger(it.src_index) ? it.src_index : k;
        const s = sent[si];
        if (!s || preds.has(s.uid)) { unbound++; return; }
        if (typeof it.speaker === 'string' && it.speaker.trim() !== s.speaker) spkMismatch++;
        preds.set(s.uid, { res: it, snap: { speaker: s.speaker, start: s.start, end: s.end, text: s.text, slot: state.slots[s.speaker] } });
      });
      state.preds = preds; state.results = res; state.resultCsv = res.csv || '';

      const warns = [];
      if (items.length !== sent.length) warns.push(`máy chủ trả ${items.length}/${sent.length} dòng (các dòng còn lại hiện là "chưa có dự đoán")`);
      if (unbound) warns.push(`${unbound} kết quả không gắn được vào dòng nào`);
      if (spkMismatch) warns.push(`${spkMismatch} dòng có speaker khác giữa yêu cầu và phản hồi`);
      if (res.csv && countCsvRecords(res.csv) - 1 !== items.length) warns.push('CSV máy chủ có số dòng khác số kết quả');
      if (warns.length) toast('Kiểm tra căn chỉnh: ' + warns.join('; ') + '.', 'warn');

      if (state.meld) recordRun();
      renderAll();
      setBusy(false, 'XONG');
      syncStep();
    } catch (e) { toast(e.message); setBusy(false, 'LỖI'); }
  }
  $('predictBtn').addEventListener('click', predict);

  /* ---------- MELD mode: toggle + preview ---------- */
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
      Object.entries(counts).forEach(([k, v]) => pills.push(el('span', 'pill', `${names[k] || k}: ${v}`)));
      box.replaceChildren(...pills); box.classList.remove('hidden');
    } catch (e) { box.classList.add('hidden'); toast(e.message); }
  }
  $('meldLoadBtn').addEventListener('click', previewMeld);

  /* ---------- run history (localStorage, latest run per dialogue) ---------- */
  const RUNS_KEY = 'digemo_meld_runs';
  const loadRuns = () => { try { return JSON.parse(localStorage.getItem(RUNS_KEY) || '[]'); } catch (_) { return []; } };
  const saveRuns = (runs) => { try { localStorage.setItem(RUNS_KEY, JSON.stringify(runs.slice(-40))); } catch (_) { } };
  const fresh = () => current().filter(c => c.pred && !c.stale);
  function recordRun() {
    const graded = fresh().filter(c => (c.row.truth || c.pred.res.truth) && typeof c.pred.res.correct === 'boolean');
    if (!graded.length) return;
    const correct = graded.filter(c => c.pred.res.correct).length;
    const runs = loadRuns().filter(r => r.dia !== state.meldDia);
    runs.push({ dia: state.meldDia, correct, graded: graded.length, ts: Date.now() });
    saveRuns(runs);
  }
  $('clearRunsBtn').addEventListener('click', () => { saveRuns([]); renderReport(); });

  /* ---------- charts ---------- */
  const NS = 'http://www.w3.org/2000/svg';
  function drawDonut(svg, segs) {
    const C = 2 * Math.PI * 48;
    const total = segs.reduce((a, s) => a + s.value, 0);
    const ring = (color, len, off) => {
      const c = document.createElementNS(NS, 'circle');
      c.setAttribute('cx', 60); c.setAttribute('cy', 60); c.setAttribute('r', 48);
      c.setAttribute('fill', 'none'); c.setAttribute('stroke-width', 16);
      c.style.stroke = color;
      if (len !== undefined) { c.setAttribute('stroke-dasharray', `${len} ${C - len}`); c.setAttribute('stroke-dashoffset', -off); }
      c.setAttribute('transform', 'rotate(-90 60 60)');
      return c;
    };
    const nodes = [ring('var(--surface-3)')];
    let acc = 0;
    if (total > 0) segs.filter(s => s.value > 0).forEach(s => { const len = s.value / total * C; nodes.push(ring(s.color, len, acc)); acc += len; });
    svg.replaceChildren(...nodes);
  }

  // Tooltip
  const tip = $('chartTip');
  function showTip(x, y, nodes) {
    tip.replaceChildren(...nodes); tip.classList.remove('hidden');
    const r = tip.getBoundingClientRect();
    let px = x + 14, py = y + 14;
    if (px + r.width > innerWidth - 8) px = Math.max(8, x - r.width - 14);
    if (py + r.height > innerHeight - 8) py = Math.max(8, y - r.height - 14);
    tip.style.left = `${px}px`; tip.style.top = `${py}px`;
  }
  const hideTip = () => tip.classList.add('hidden');

  function renderAnalytics() {
    const card = $('analyticsCard');
    const cur = current();
    const preds = cur.filter(c => c.pred);
    if (!preds.length) { card.classList.add('hidden'); return; }
    card.classList.remove('hidden');

    // A. Overview (only figures that exist in the data)
    const counts = Object.fromEntries(EMO.map(l => [l, 0]));
    preds.forEach(c => { if (c.pred.res.emotion in counts) counts[c.pred.res.emotion]++; });
    const confs = preds.map(c => Number(c.pred.res.confidence)).filter(Number.isFinite);
    const avg = confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : null;
    const top = EMO.reduce((a, b) => (counts[b] > counts[a] ? b : a), EMO[0]);
    const f = fresh().filter(c => (c.row.truth || c.pred.res.truth) && typeof c.pred.res.correct === 'boolean');
    const stat = (label, value, cls, strongCls) => { const s = el('div', `stat ${cls || ''}`.trim()); s.append(el('small', '', label), el('strong', strongCls || '', value)); return s; };
    const stats = [
      stat('Tổng utterance', String(state.rows.length)),
      stat('Đã dự đoán', `${preds.length}/${state.rows.length}`),
      stat('Số người nói', String(speakerNames().length)),
      avg !== null ? stat('Độ tin cậy TB', `${avg.toFixed(1)}%`) : null,
      stat('Cảm xúc nổi bật', counts[top] ? names[top] : '—', `${counts[top] ? emoCls(top) : ''}`, 'emo-text'),
      f.length ? stat('Độ chính xác', `${(f.filter(c => c.pred.res.correct).length / f.length * 100).toFixed(1)}%`, 'good') : null
    ].filter(Boolean);
    $('ovStats').replaceChildren(...stats);
    const nStale = preds.filter(c => c.stale).length;
    $('ovNote').textContent = nStale ? `${nStale} DÒNG ĐÃ SỬA SAU DỰ ĐOÁN` : '';

    // B. Distribution
    const total = EMO.reduce((a, l) => a + counts[l], 0);
    drawDonut($('distDonut'), EMO.map(l => ({ value: counts[l], color: `var(--emo-${l})` })));
    $('distTotal').textContent = String(total);
    $('distLegend').replaceChildren(...EMO.map(l => {
      const row = el('div', `legend-row ${emoCls(l)}${counts[l] ? '' : ' zero'}`);
      row.append(el('i'), el('span', 'nm', names[l]), el('span', 'ct', String(counts[l])), el('span', 'pc', total ? `${(counts[l] / total * 100).toFixed(0)}%` : '—'));
      return row;
    }));
    renderTimeline();
  }

  function renderTimeline() {
    const chart = $('tlChart'), axis = $('tlAxis');
    const sp = state.tlSpeaker;
    const list = current().filter(c => c.pred && (!sp || c.pred.snap.speaker === sp));
    hideTip();
    if (!list.length) { chart.replaceChildren(el('div', 'run-empty', 'Không có dữ liệu cho speaker này.')); axis.replaceChildren(); }
    else {
      chart.replaceChildren(...list.map((c, k) => {
        const it = c.pred.res, s = c.pred.snap, conf = Number(it.confidence);
        const b = el('button', `tl-bar ${emoCls(it.emotion)}${c.stale ? ' stale' : ''}`); b.type = 'button';
        b.style.height = `${Math.max(4, Math.min(100, Number.isFinite(conf) ? conf : 0))}%`;
        b.setAttribute('aria-label', `Câu ${c.i + 1}, ${s.speaker}, ${names[it.emotion] || it.emotion}, ${Number.isFinite(conf) ? conf.toFixed(1) : '?'}%`);
        const content = () => {
          const head = el('b', '', `#${c.i + 1} · ${s.speaker}`);
          const t = el('span', '', `${fmt(s.start)} – ${fmt(s.end)}`);
          const e = el('span', '', `${names[it.emotion] || it.emotion} · ${Number.isFinite(conf) ? conf.toFixed(1) + '%' : '—'}${c.stale ? ' (dự đoán cũ)' : ''}`);
          const tx = el('span', 'tip-text', s.text.length > 140 ? s.text.slice(0, 140) + '…' : s.text);
          return [head, document.createElement('br'), t, document.createElement('br'), e, tx];
        };
        b.addEventListener('mouseenter', (ev) => showTip(ev.clientX, ev.clientY, content()));
        b.addEventListener('mousemove', (ev) => showTip(ev.clientX, ev.clientY, content()));
        b.addEventListener('mouseleave', hideTip);
        b.addEventListener('focus', () => { const r = b.getBoundingClientRect(); showTip(r.left, r.top - 90, content()); });
        b.addEventListener('blur', hideTip);
        b.addEventListener('click', () => playRow(c.row.uid));
        return b;
      }));
      axis.replaceChildren(...list.map((c, k) => el('span', '', k % 5 === 0 ? String(c.i + 1) : '')));
    }
    $('tlLegend').replaceChildren(...EMO.map(l => { const s = el('span', emoCls(l)); s.append(el('i'), document.createTextNode(names[l])); return s; }));
  }

  /* ---------- MELD report ---------- */
  function renderReport() {
    const card = $('reportCard');
    if (!state.meld || !state.preds.size) { card.classList.add('hidden'); return; }
    const items = fresh();
    const graded = items.filter(c => (c.row.truth || c.pred.res.truth) && typeof c.pred.res.correct === 'boolean');
    if (!items.length) { card.classList.add('hidden'); return; }
    const truthOf = (c) => c.row.truth || c.pred.res.truth;
    const ok = graded.filter(c => c.pred.res.correct).length;
    const bad = graded.length - ok;
    const acc = graded.length ? ok / graded.length : 0;
    card.classList.remove('hidden');
    $('reportDia').textContent = `#${state.meldDia}`;
    $('reportTag').textContent = graded.length ? `${ok}/${graded.length} CÂU ĐÚNG` : 'KHÔNG CÓ NHÃN';

    drawDonut($('donut'), graded.length ? [{ value: ok, color: 'var(--success)' }, { value: bad, color: 'var(--error)' }] : []);
    $('donutPct').textContent = graded.length ? `${(acc * 100).toFixed(0)}%` : '—';
    $('reportStats').replaceChildren(
      ...[['Câu có nhãn', graded.length, ''], ['Dự đoán đúng', ok, 'good'], ['Dự đoán sai', bad, 'bad']].map(([t, v, cls]) => {
        const s = el('div', `stat ${cls}`.trim()); s.append(el('small', '', t), el('strong', '', String(v))); return s;
      }));

    $('timeline').replaceChildren(...items.map(c => {
      const t = truthOf(c), r = c.pred.res;
      const cls = !t || typeof r.correct !== 'boolean' ? 'na' : r.correct ? 'ok' : 'no';
      const cell = el('div', `tl-cell ${cls}`);
      cell.append(el('span', '', cls === 'na' ? '·' : cls === 'ok' ? '✓' : '✗'), document.createTextNode(String(c.i + 1)));
      cell.title = `#${c.i + 1} ${c.pred.snap.speaker}: model = ${names[r.emotion] || r.emotion}` + (t ? ` · CSV = ${names[t] || t}` : '');
      return cell;
    }));

    const per = Object.fromEntries(EMO.map(l => [l, { ok: 0, no: 0 }]));
    graded.forEach(c => { const t = truthOf(c); if (per[t]) per[t][c.pred.res.correct ? 'ok' : 'no']++; });
    $('emoBars').replaceChildren(...EMO.map(l => {
      const { ok: a, no: b } = per[l], n = a + b;
      const row = el('div', 'emo-row'); row.append(el('span', 'nm', names[l]));
      const track = el('div', 'emo-track');
      if (n) { const g = el('i', 'g'); g.style.width = `${a / n * 100}%`; const r = el('i', 'r'); r.style.width = `${b / n * 100}%`; track.append(g, r); }
      row.append(track, el('span', 'ct', n ? `${a}/${n}` : '—'));
      return row;
    }));

    const idx = Object.fromEntries(EMO.map((l, i) => [l, i]));
    const M = EMO.map(() => EMO.map(() => 0));
    graded.forEach(c => { const t = truthOf(c); if (t in idx && c.pred.res.emotion in idx) M[idx[t]][idx[c.pred.res.emotion]]++; });
    const cells = [el('span', 'h', 'THẬT↓')];
    EMO.forEach(l => cells.push(el('span', 'h', short(l))));
    M.forEach((rowv, i) => {
      const tot = rowv.reduce((a, b) => a + b, 0);
      cells.push(el('span', 'h', short(EMO[i])));
      rowv.forEach((v, j) => {
        const c = el('span', `c${i === j ? ' diag' : ''}`, v ? String(v) : '');
        if (v) {
          const fr = v / tot;
          c.style.background = i === j ? `rgba(15,118,110,${0.15 + 0.85 * fr})` : `rgba(220,38,38,${0.12 + 0.88 * fr})`;
          if (fr > 0.55) c.classList.add('hit');
        }
        c.title = `Thật: ${names[EMO[i]]} → Model: ${names[EMO[j]]} (${v})`;
        cells.push(c);
      });
    });
    const conf = el('div', 'conf'); conf.append(...cells);
    $('confMatrix').replaceChildren(conf);

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
  syncStep();
})();