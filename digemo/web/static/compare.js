/* Tab "So sánh model": DiGEmo vs DM-CER trên tập test (MELD / IEMOCAP).
   Dữ liệu từ /api/compare/* (digemo/web/compare_routes.py -> scripts/compare_models.py). */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const MODELS = [['DiGEmo', 'a'], ['DM-CER', 'b']];
  // màu chip cảm xúc (tokens.css); IEMOCAP dùng tên ngắn
  const EMO_CLASS = {
    neutral: 'neutral', surprise: 'surprise', fear: 'fear', sadness: 'sadness', joy: 'joy', disgust: 'disgust',
    anger: 'anger', neu: 'neutral', sad: 'sadness', ang: 'anger', hap: 'joy', exc: 'surprise', fru: 'fear',
  };
  const CAT_TEXT = { only_digemo: 'Chỉ DiGEmo đúng', only_dmcer: 'Chỉ DM-CER đúng', both_wrong: 'Cả hai sai' };

  const st = { ds: 'meld', cfg: null, res: null, filter: 'all', limit: 100 };

  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = (v, d = 2) => (v == null || Number.isNaN(v) ? '—' : (100 * v).toFixed(d));
  const chip = l => `<span class="emotion-chip emo-${EMO_CLASS[l] || 'neutral'}">${esc(l ?? '—')}</span>`;

  // ---------------------------------------------------------------- toast / status
  let toastTimer;
  function toast(msg, kind = '') {
    $('toastText').textContent = msg;
    $('toast').className = `toast ${kind}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $('toast').classList.add('hidden'), 6000);
  }
  $('toastClose').onclick = () => $('toast').classList.add('hidden');

  function busy(on, text) {
    $('runBtn').disabled = on;
    $('loadBtn').disabled = on;
    document.querySelectorAll('#dsChips .chip').forEach(b => { b.disabled = on; });
    $('progress').classList.toggle('hidden', !on);
    if (text) $('progressText').textContent = text;
    $('appStatus').classList.toggle('busy', on);
    $('appStatusText').textContent = on ? 'Đang chạy…' : 'Sẵn sàng';
    if (on) {
      $('runTag').className = 'tag busy';
      $('runTag').textContent = 'ĐANG CHẠY';
    }
  }

  async function api(url, opts) {
    const r = await fetch(url, opts);
    let body = null;
    try { body = await r.json(); } catch (_) { /* không phải json */ }
    if (!r.ok) {
      const err = new Error((body && body.detail) || `${r.status} ${r.statusText}`);
      err.status = r.status;
      throw err;
    }
    return body;
  }

  // ---------------------------------------------------------------- config
  const CFG_ROWS = [
    ['pkl', 'pkl gốc (nhãn thật)', true],
    ['csv', 'test_sent_emo.csv (ghép câu)', false],
    ['key_map', 'key_map.json', false],
    ['digemo_ckpt', 'Checkpoint DiGEmo', true],
    ['digemo_pred', 'Dự đoán DiGEmo có sẵn', false],
    ['dmcer_pred', 'File dự đoán DM-CER', true],
    ['dmcer_code_dir', 'Thư mục code DM-CER', false],
    ['dmcer_data_root', 'data_root DM-CER', false],
    ['dmcer_ckpt', 'Checkpoint DM-CER', false],
  ];

  function renderConfig() {
    const c = st.cfg && st.cfg[st.ds];
    if (!c) return;
    $('cfgFile').textContent = st.cfg.config_file + (st.cfg.config_exists ? '' : ' (chưa có, đang dùng mặc định)');
    $('cfgList').innerHTML = CFG_ROWS.filter(([k]) => !(st.ds === 'iemocap' && (k === 'key_map' || k === 'csv'))).map(([k, name, need]) => {
      const p = c[k] || {};
      const mark = !p.path ? '<span class="opt-k">bỏ trống</span>'
        : p.exists ? '<span class="ok">✓ có</span>'
          : `<span class="${need ? 'no' : 'opt-k'}">✗ không thấy</span>`;
      return `<span class="k">${esc(name)}</span><code>${esc(p.path || '—')}</code>${mark}`;
    }).join('') + `<span class="k">Thư mục kết quả</span><code>${esc(st.cfg.out_dir)}\\${st.ds}</code><span></span>`;
    $('runDmcer').disabled = !c.can_run_dmcer;
    if (!c.can_run_dmcer) $('runDmcer').checked = false;
    $('dmcerHint').textContent = c.can_run_dmcer ? '(gọi inference.py, mất vài phút)'
      : '(cần dmcer_code_dir, dmcer_data_root, dmcer_ckpt trong config)';
    $('loadBtn').disabled = !(st.cfg.has_result || {})[st.ds];
    const missing = [];
    if (!c.can_run_digemo) missing.push('pkl hoặc checkpoint DiGEmo');
    if (!c.dmcer_pred.exists && !c.can_run_dmcer) missing.push('file dự đoán DM-CER');
    if (missing.length) {
      $('cfgBox').open = true;
      showError(`Thiếu ${missing.join(' và ')} cho ${st.ds.toUpperCase()}. Sửa ${st.cfg.config_file} rồi tải lại trang.`);
    } else {
      showError('');
    }
  }

  async function loadConfig() {
    try {
      st.cfg = await api('/api/compare/config');
      renderConfig();
    } catch (e) {
      showError(`Không đọc được cấu hình: ${e.message}`);
    }
  }

  function showError(msg) {
    $('errBox').textContent = msg;
    $('errBox').classList.toggle('hidden', !msg);
  }

  // ---------------------------------------------------------------- run / load
  async function run() {
    showError('');
    busy(true, $('runDmcer').checked ? 'Đang chạy inference DM-CER rồi DiGEmo trên tập test… (vài phút)'
      : 'Đang chạy DiGEmo trên tập test và đọc dự đoán DM-CER…');
    try {
      const res = await api('/api/compare/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: st.ds, run_dmcer: $('runDmcer').checked }),
      });
      show(res);
      toast(`Xong ${st.ds.toUpperCase()}: ${res.coverage.common} câu, ${res.seconds}s`, 'ok');
      loadConfig();
    } catch (e) {
      showError(e.message);
      toast('Chạy so sánh lỗi, xem chi tiết bên dưới nút.', 'warn');
      $('runTag').className = 'tag';
      $('runTag').textContent = 'LỖI';
    } finally {
      busy(false);
    }
  }

  async function loadSaved(quiet) {
    try {
      show(await api(`/api/compare/result/${st.ds}`));
    } catch (e) {
      if (!quiet) toast(e.message, 'warn');
      hideResults();
    }
  }

  function hideResults() {
    st.res = null;
    ['resultCard', 'disCard', 'logCard'].forEach(id => $(id).classList.add('hidden'));
    $('runTag').className = 'tag';
    $('runTag').textContent = 'CHƯA CHẠY';
  }

  // ---------------------------------------------------------------- render
  function show(res) {
    st.res = res;
    st.limit = 100;
    $('runTag').className = 'tag ready';
    $('runTag').textContent = res.dataset.toUpperCase();
    $('resKicker').textContent = `KẾT QUẢ ${res.dataset.toUpperCase()} · TEST`;
    $('resN').textContent = `${res.coverage.common} câu`;
    $('resTime').textContent = `CHẠY LÚC ${res.created || ''}`;
    renderMetrics(res);
    renderAgree(res);
    renderF1(res);
    renderCM(res);
    renderDownloads(res);
    setupDisFilters(res);
    renderDis();
    $('logBox').textContent = [res.log || '', res.dmcer_log ? `--- DM-CER ---\n${res.dmcer_log}` : ''].join('\n').trim()
      || 'Không có log (kết quả đọc từ file đã lưu).';
    ['resultCard', 'disCard', 'logCard'].forEach(id => $(id).classList.remove('hidden'));
  }

  function renderMetrics(res) {
    const A = res.models['DiGEmo'], B = res.models['DM-CER'];
    const rows = [['Accuracy', A.accuracy, B.accuracy], ['Weighted F1', A.weighted_f1, B.weighted_f1],
      ['Macro F1', A.macro_f1, B.macro_f1]];
    res.labels.forEach(l => rows.push([`F1 ${l}`, A.per_class[l].f1, B.per_class[l].f1, A.per_class[l].support]));
    const cell = (v, o) => `<td class="${v > o + 1e-12 ? 'best' : ''}">${pct(v)}</td>`;
    $('metricTable').innerHTML = `<thead><tr><th>Chỉ số (%)</th><th><i class="mdot a"></i>DiGEmo</th>
      <th><i class="mdot b"></i>DM-CER</th><th>Δ (DM-CER − DiGEmo)</th><th>Số câu thật</th></tr></thead><tbody>` +
      rows.map(([n, a, b, sup], i) => {
        const d = 100 * (b - a);
        const cls = Math.abs(d) < 0.005 ? 'zero' : d > 0 ? 'pos' : 'neg';
        return `<tr${i === 2 ? ' style="border-bottom:2px solid var(--border-2)"' : ''}><td>${i > 2 ? chip(n.slice(3)) : `<strong>${n}</strong>`}</td>
          ${cell(a, b)}${cell(b, a)}<td class="delta ${cls}">${d > 0 ? '+' : ''}${d.toFixed(2)}</td><td>${sup ?? res.coverage.common}</td></tr>`;
      }).join('') + '</tbody>';
    const c = res.coverage;
    const notes = [`pkl test ${c.truth} câu · DiGEmo dự đoán ${c.digemo} · DM-CER dự đoán ${c.dmcer} · tính chỉ số trên ${c.common} câu chung.`];
    if (c.truth - c.dmcer > 0) notes.push(`${c.truth - c.dmcer} câu không có dự đoán DM-CER (bị bỏ khi tiền xử lý hoặc khác khóa).`);
    if (c.dmcer_extra_keys) notes.push(`${c.dmcer_extra_keys} dòng trong file DM-CER không có trong pkl.`);
    if (c.label_mismatch_vs_dmcer_file) notes.push(`Cảnh báo: ${c.label_mismatch_vs_dmcer_file} câu có nhãn trong file DM-CER khác nhãn pkl (dùng nhãn pkl).`);
    const fc = res.full_coverage || {};
    if (fc['DM-CER'] && fc['DM-CER'].n !== c.common) {
      notes.push(`DM-CER trên toàn bộ ${fc['DM-CER'].n} câu của nó: ACC ${pct(fc['DM-CER'].accuracy)}, W-F1 ${pct(fc['DM-CER'].weighted_f1)}.`);
    }
    $('coverageNote').textContent = notes.join(' ');
  }

  function renderAgree(res) {
    const g = res.agreement, n = res.coverage.common;
    const tile = (label, v, cls = '') => `<div class="stat ${cls}"><small>${label}</small><strong>${v}</strong>
      <small style="margin:4px 0 0">${(100 * v / n).toFixed(1)}%</small></div>`;
    $('agreeStats').innerHTML = tile('Cả hai đúng', g.both_correct, 'good') +
      tile('<i class="mdot a"></i>Chỉ DiGEmo đúng', g.only_digemo) +
      tile('<i class="mdot b"></i>Chỉ DM-CER đúng', g.only_dmcer) +
      tile('Cả hai sai', g.both_wrong, 'bad') +
      `<div class="stat"><small>Hai model trùng dự đoán</small><strong>${pct(g.agree_rate, 1)}%</strong>
       <small style="margin:4px 0 0">${g.both_wrong_same_label} câu sai giống nhau</small></div>`;
    const p = g.mcnemar.p;
    $('mcnemar').textContent = `McNEMAR χ² ${g.mcnemar.chi2.toFixed(2)} · p ${p < 0.001 ? '< 0.001' : p.toFixed(3)}` +
      (p < 0.05 ? ' · KHÁC BIỆT CÓ Ý NGHĨA' : ' · CHƯA CÓ Ý NGHĨA (p ≥ 0.05)');
  }

  function renderF1(res) {
    $('f1Bars').innerHTML = res.labels.map(l => {
      const sup = res.models['DiGEmo'].per_class[l].support;
      const lines = MODELS.map(([m, k]) => {
        const pc = res.models[m].per_class[l];
        const tip = `${m} · ${l}: F1 ${pct(pc.f1)} · P ${pct(pc.precision)} · R ${pct(pc.recall)}`;
        return `<div class="f1-line"><div class="f1-track ${k}" data-tip="${esc(tip)}"><i style="width:${(100 * pc.f1).toFixed(1)}%"></i></div>
          <span>${pct(pc.f1, 1)}</span></div>`;
      }).join('');
      return `<div class="f1-row"><div class="lbl">${chip(l)}<small>${sup} câu</small></div><div class="f1-pair">${lines}</div></div>`;
    }).join('');
  }

  function renderCM(res) {
    const L = res.labels, mode = $('cmMode').value, K = L.length;
    $('cmPair').innerHTML = MODELS.map(([m, k]) => {
      const cm = res.models[m].confusion;
      let html = `<div class="cm-grid" style="grid-template-columns:64px repeat(${K}, minmax(30px,1fr))"><span></span>` +
        L.map(l => `<span class="h" title="${esc(l)}">${esc(l.slice(0, 4))}</span>`).join('');
      cm.forEach((row, i) => {
        const tot = row.reduce((s, v) => s + v, 0) || 1;
        html += `<span class="h row" title="${esc(L[i])}">${esc(L[i])}</span>`;
        row.forEach((v, j) => {
          const f = v / tot;
          const shade = Math.round(8 + 82 * f);
          const bg = v ? `background:color-mix(in srgb, var(--model-${k}) ${shade}%, var(--surface-1))` : '';
          const txt = mode === 'pct' ? (v ? (100 * f).toFixed(0) : '·') : v;
          const tip = `${m} · thật ${L[i]} → dự đoán ${L[j]}: ${v} câu (${(100 * f).toFixed(1)}% của ${L[i]})`;
          html += `<span class="c${i === j ? ' diag' : ''}${shade > 55 ? ' dark' : ''}" style="${bg}" data-tip="${esc(tip)}">${txt}</span>`;
        });
      });
      html += '</div>';
      const M = res.models[m];
      return `<div><div class="cm-title"><span><i class="mdot ${k}"></i>${m}</span>
        <small>ACC ${pct(M.accuracy)} · W-F1 ${pct(M.weighted_f1)}</small></div>
        <div class="conf-wrap">${html}</div><div class="cm-axis">HÀNG = NHÃN THẬT · CỘT = DỰ ĐOÁN</div></div>`;
    }).join('');
  }

  function renderDownloads(res) {
    const files = [['per_utterance.csv', 'Từng câu (csv)'], ['disagreements.csv', 'Câu sai khác (csv)'],
      ['metrics.json', 'Metrics (json)'], ['summary.md', 'Bảng tóm tắt (md)'],
      ['cm_compare.png', 'Ma trận nhầm lẫn (png)'], ['f1_per_class.png', 'F1 từng lớp (png)']];
    $('downloads').innerHTML = files.map(([f, n]) =>
      `<a class="btn btn-outline btn-sm" href="/api/compare/file/${res.dataset}/${f}" download>
        <svg class="icon"><use href="#i-download" /></svg><span>${n}</span></a>`).join('');
  }

  // ---------------------------------------------------------------- disagreement table
  function setupDisFilters(res) {
    $('disTrue').innerHTML = '<option value="">Mọi nhãn thật</option>' +
      res.labels.map(l => `<option value="${esc(l)}">Nhãn thật: ${esc(l)}</option>`).join('');
  }

  function disRows() {
    const res = st.res;
    if (!res) return [];
    const q = $('disSearch').value.trim().toLowerCase();
    const tl = $('disTrue').value;
    return res.rows.filter(r => r.category !== 'missing' && !r.agree &&
      (st.filter === 'all' || r.category === st.filter) && (!tl || r.true === tl) &&
      (!q || `${r.key} ${r.text} ${r.speaker}`.toLowerCase().includes(q)));
  }

  function renderDis() {
    const all = st.res.rows.filter(r => r.category !== 'missing' && !r.agree);
    $('disN').textContent = all.length;
    const rows = disRows();
    const shown = rows.slice(0, st.limit);
    const mark = ok => ok ? '<span class="ok-mark" aria-label="đúng">✓</span>' : '<span class="no-mark" aria-label="sai">✗</span>';
    $('disTable').innerHTML = `<thead><tr><th>Câu</th><th>Text</th><th>Nhãn thật</th>
      <th><i class="mdot a"></i>DiGEmo</th><th><i class="mdot b"></i>DM-CER</th><th>Kết luận</th></tr></thead><tbody>` +
      (shown.length ? shown.map(r => `<tr>
        <td class="mono">${esc(r.key)}${r.speaker ? `<br>${esc(r.speaker)}` : ''}</td>
        <td class="txt">${esc(r.text)}</td>
        <td>${chip(r.true)}</td>
        <td>${chip(r.digemo)}${mark(r.digemo_ok)}${r.digemo_conf != null ? `<br><small class="hint">${(100 * r.digemo_conf).toFixed(0)}%</small>` : ''}</td>
        <td>${chip(r.dmcer)}${mark(r.dmcer_ok)}</td>
        <td><span class="verdict ${r.category}">${CAT_TEXT[r.category] || r.category}</span></td></tr>`).join('')
        : '<tr><td colspan="6" style="text-align:center;color:var(--muted)">Không có câu nào khớp bộ lọc</td></tr>') + '</tbody>';
    $('disShown').textContent = `Hiện ${shown.length} / ${rows.length} câu`;
    $('disMore').classList.toggle('hidden', rows.length <= st.limit);
  }

  // ---------------------------------------------------------------- tooltip
  const tip = $('chartTip');
  document.addEventListener('mouseover', e => {
    const t = e.target.closest('[data-tip]');
    if (!t) { tip.classList.add('hidden'); return; }
    tip.textContent = t.dataset.tip;
    tip.classList.remove('hidden');
  });
  document.addEventListener('mousemove', e => {
    if (tip.classList.contains('hidden')) return;
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = `${x}px`;
    tip.style.top = `${e.clientY + 14}px`;
  });

  // ---------------------------------------------------------------- events
  $('dsChips').addEventListener('click', e => {
    const b = e.target.closest('.chip');
    if (!b || b.dataset.ds === st.ds) return;
    st.ds = b.dataset.ds;
    document.querySelectorAll('#dsChips .chip').forEach(c => {
      const on = c === b;
      c.classList.toggle('active', on);
      c.setAttribute('aria-checked', on);
    });
    renderConfig();
    loadSaved(true);
  });
  $('runBtn').onclick = run;
  $('loadBtn').onclick = () => loadSaved(false);
  $('cmMode').onchange = () => st.res && renderCM(st.res);
  $('disChips').addEventListener('click', e => {
    const b = e.target.closest('.chip');
    if (!b) return;
    st.filter = b.dataset.f;
    st.limit = 100;
    document.querySelectorAll('#disChips .chip').forEach(c => c.classList.toggle('active', c === b));
    renderDis();
  });
  $('disTrue').onchange = () => { st.limit = 100; renderDis(); };
  $('disSearch').oninput = () => { st.limit = 100; renderDis(); };
  $('disMore').onclick = () => { st.limit += 100; renderDis(); };
  window.addEventListener('scroll', () => $('topbar').classList.toggle('scrolled', window.scrollY > 10), { passive: true });

  loadConfig().then(() => loadSaved(true));
})();
