/* Buildless, browser-local interactions. No analytics, API calls or dependencies. */
(() => {
  'use strict';
  const byId = (id) => document.getElementById(id);
  const all = (selector) => [...document.querySelectorAll(selector)];

  // Inputs are illustrative percentiles, not statistics inferred from robot data.
  const presets = {
    calm: [35, 25],
    temporal: [93, 40],
    cross: [30, 95],
    both: [92, 94],
  };
  const temporal = byId('temporal-range');
  const cross = byId('cross-range');
  const threshold = byId('threshold-range');
  const paintRange = (input) => {
    const percent = 100 * (Number(input.value) - Number(input.min)) / (Number(input.max) - Number(input.min));
    input.style.setProperty('--fill', `${percent}%`);
  };
  const decimal = (value) => (value / 100).toFixed(2);

  function updateTrigger() {
    const t = Number(temporal.value);
    const c = Number(cross.value);
    const limit = Number(threshold.value);
    const score = Math.max(t, c);
    // Paper Eq. 7: strictly greater, either axis is sufficient.
    const activated = score > limit;
    byId('temporal-value').textContent = decimal(t);
    byId('cross-value').textContent = decimal(c);
    byId('threshold-value').textContent = decimal(limit);
    [temporal, cross, threshold].forEach(paintRange);

    const boundaryX = 55 + 3.4 * limit;
    const boundaryY = 290 - 2.7 * limit;
    byId('trigger-region').setAttribute('d', `M55 20H395V290H${boundaryX}V${boundaryY}H55Z`);
    byId('threshold-lines').setAttribute('d', `M55 ${boundaryY}H395M${boundaryX} 20V290`);
    const pointX = 55 + 3.4 * t;
    const pointY = 290 - 2.7 * c;
    for (const id of ['trigger-point', 'trigger-point-halo']) {
      byId(id).setAttribute('cx', pointX);
      byId(id).setAttribute('cy', pointY);
      byId(id).setAttribute('fill', activated ? '#d84c35' : '#446a52');
    }
    const pointLabel = byId('trigger-point-label');
    pointLabel.textContent = `(${decimal(t)}, ${decimal(c)})`;
    pointLabel.setAttribute('x', Math.min(290, Math.max(65, pointX - 43)));
    pointLabel.setAttribute('y', pointY < 60 ? pointY + 25 : pointY - 18);
    byId('trigger-decision').classList.toggle('inactive', !activated);
    byId('decision-title').textContent = activated ? '启动验证' : '直接执行';
    byId('decision-budget').textContent = activated ? '2 → 10 个候选' : '采用第 1 个候选';
    byId('decision-detail').textContent = `max(${decimal(t)}, ${decimal(c)}) = ${decimal(score)} ${activated ? '>' : '≤'} ${decimal(limit)}`;
    all('[data-preset]').forEach((button) => {
      const values = presets[button.dataset.preset];
      const selected = values[0] === t && values[1] === c;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
  }
  [temporal, cross, threshold].forEach((input) => input.addEventListener('input', updateTrigger));
  all('[data-preset]').forEach((button) => button.addEventListener('click', () => {
    [temporal.value, cross.value] = presets[button.dataset.preset];
    updateTrigger();
  }));
  updateTrigger();

  // Transcribed from paper Tables 1–3. Reported means retain paper rounding.
  const benchmarks = {
    real: {
      metric: '平均完成得分 / %', title: '精细操作中的提升最直接。',
      note: 'π₀.₅ 基础策略；3 项任务，每种方法每项任务 50 次试验。按阶段计分，部分完成也能获得分数。',
      baseline: 50.8, ours: 65.5,
      rows: [['插笔', 45, 60], ['移液', 52.5, 63.5], ['苹果放置', 55, 73]],
    },
    clean: {
      metric: '平均成功率 / % · 始终开启验证', title: '五项精细双臂任务，均有提升。',
      note: '基础策略为 Hy-Embodied-0.5-VLA。这组表 2 结果使用始终开启验证；自适应触发的权衡另见论文图 6。',
      baseline: 66.4, ours: 73.6,
      rows: [['Mug', 38, 48], ['Switch', 44, 50], ['Seal', 74, 82], ['Mouse', 88, 96], ['Scan', 88, 92]],
    },
    random: {
      metric: '平均成功率 / % · 始终开启验证', title: '从干净训练，走向随机化测试。',
      note: '策略和验证器均仅使用 clean 数据训练；在随机化环境中测试，始终开启验证。基础策略为 Hy-Embodied-0.5-VLA。',
      baseline: 62.8, ours: 69.6,
      rows: [['Mug', 16, 26], ['Switch', 50, 56], ['Seal', 72, 78], ['Mouse', 80, 92], ['Scan', 96, 96]],
    },
    libero: {
      metric: '平均成功率 / % · π₀.₅ 基础策略', title: '高基线上，增益集中在未饱和任务。',
      note: '4 个套件共 40 项任务；LIBERO-10 从 92.4% 提升至 94.0%。StarVLA-GR00T 的另一组结果使用独立验证器权重。',
      baseline: 96.9, ours: 97.6,
      rows: [['Spatial', 98.8, 98.8], ['Object', 98.2, 98.8], ['Goal', 98, 98.6], ['LIBERO-10', 92.4, 94]],
    },
  };
  function updateBenchmark(key) {
    const data = benchmarks[key];
    byId('results-metric').textContent = data.metric;
    byId('results-title').textContent = data.title;
    byId('results-note').textContent = data.note;
    byId('result-baseline').textContent = data.baseline.toFixed(1);
    byId('result-ours').replaceChildren(document.createTextNode(data.ours.toFixed(1)));
    const percent = document.createElement('span');
    percent.textContent = '%';
    byId('result-ours').append(percent);
    byId('result-gain').textContent = `+${(data.ours - data.baseline).toFixed(1)} 个百分点`;
    byId('results-panel').setAttribute('aria-labelledby', `tab-${key}`);
    const fragment = document.createDocumentFragment();
    data.rows.forEach(([label, base, ours]) => {
      const row = document.createElement('div');
      row.className = 'result-bar-row';
      row.setAttribute('aria-label', `${label}：基础策略 ${base}%，VeriVLA ${ours}%`);
      const name = document.createElement('span');
      name.className = 'result-bar-label';
      name.textContent = label;
      const pair = document.createElement('div');
      pair.className = 'bar-pair';
      pair.setAttribute('aria-hidden', 'true');
      [base, ours].forEach((value, index) => {
        const track = document.createElement('div');
        track.className = `bar-track ${index ? 'ours' : 'baseline'}`;
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.style.width = `${value}%`;
        const caption = document.createElement('span');
        caption.textContent = value.toFixed(1);
        bar.append(caption);
        track.append(bar);
        pair.append(track);
      });
      row.append(name, pair);
      fragment.append(row);
    });
    byId('result-bars').replaceChildren(fragment);
    all('[data-benchmark]').forEach((tab) => {
      const selected = tab.dataset.benchmark === key;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
  }
  const tabs = all('[data-benchmark]');
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => updateBenchmark(tab.dataset.benchmark));
    tab.addEventListener('keydown', (event) => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        tabs[next].focus();
        updateBenchmark(tabs[next].dataset.benchmark);
      }
    });
  });
  updateBenchmark('real');

  const rate = byId('rate-range');
  function updateLatency() {
    const r = Number(rate.value) / 100;
    byId('rate-value').textContent = `${rate.value}%`;
    byId('latency-value').textContent = (96 + 326 * r).toFixed(1);
    paintRange(rate);
    const segments = { base: 82, trigger: 14, sample: 80 * r, verify: 246 * r };
    Object.entries(segments).forEach(([key, value]) => {
      byId(`latency-${key}`).style.flex = `${value} 1 0%`;
      byId(`latency-${key}`).style.display = value === 0 ? 'none' : 'block';
    });
  }
  rate.addEventListener('input', updateLatency);
  updateLatency();

  const dialog = byId('figure-dialog');
  all('[data-figure]').forEach((button) => button.addEventListener('click', () => {
    byId('dialog-image').src = button.dataset.figure;
    byId('dialog-image').alt = button.dataset.caption;
    byId('dialog-caption').textContent = button.dataset.caption;
    dialog.showModal();
  }));
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) {
      const box = dialog.getBoundingClientRect();
      if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
    }
  });
  all('.print-button').forEach((button) => button.addEventListener('click', () => window.print()));
  let printExpanded = [];
  window.addEventListener('beforeprint', () => {
    printExpanded = all('details:not([open])');
    printExpanded.forEach((details) => { details.open = true; });
  });
  window.addEventListener('afterprint', () => {
    printExpanded.forEach((details) => { details.open = false; });
    printExpanded = [];
  });

  const sections = all('main > section[id]');
  const navLinks = all('.toc a');
  let scheduled = false;
  function updateReading() {
    scheduled = false;
    const available = document.documentElement.scrollHeight - window.innerHeight;
    byId('reading-progress').style.width = `${available > 0 ? Math.min(100, 100 * window.scrollY / available) : 100}%`;
    let active = sections[0].id;
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= 160) active = section.id;
    }
    navLinks.forEach((link) => {
      const selected = link.getAttribute('href') === `#${active}`;
      link.classList.toggle('active', selected);
      if (selected) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
  window.addEventListener('scroll', () => {
    if (!scheduled) { scheduled = true; requestAnimationFrame(updateReading); }
  }, { passive: true });
  window.addEventListener('resize', updateReading, { passive: true });
  updateReading();
})();
