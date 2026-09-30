(() => {
  'use strict';
  const $ = (selector, scope = document) => scope.querySelector(selector);
  const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];
  const escape = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const state = { family: 'Qwen2.5-VL-7B', metric: 'Overview' };
  let content;

  const toc = $('.floating-toc');
  const tocToggle = $('.toc-toggle', toc);
  toc.addEventListener('click', event => {
    if (event.target.closest('.toc-panel a')) {
      toc.open = false;
      tocToggle.focus({preventScroll:true});
    }
  });
  document.addEventListener('click', event => {
    if (toc.open && !toc.contains(event.target)) toc.open = false;
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && toc.open) {
      event.preventDefault();
      toc.open = false;
      tocToggle.focus({preventScroll:true});
    }
  });

  function columnsForView() {
    return content.results.columns.filter(column => state.metric === 'Overview' ? column.id.endsWith('_avg') : column.category === state.metric);
  }

  function tableHeaders(columns, isFull = false) {
    let groupCells = '';
    const categories = [...new Set(columns.map(column => column.category))];
    for (const category of categories) {
      const indices = columns.map((column, index) => column.category === category ? index : -1).filter(index => index >= 0);
      groupCells += `<th scope="colgroup" colspan="${indices.length}" data-cols="${indices.join(',')}">${escape(category)}</th>`;
    }
    const secondRow = columns.map((column, index) => `<th scope="col" data-col="${index}">${escape(column.label)} <span aria-label="higher is better">↑</span></th>`).join('');
    return `<caption>${isFull ? 'Complete results' : escape(state.family)} — accuracy (%), higher is better</caption><thead><tr class="table-super-header"><th scope="col" rowspan="2">${isFull ? 'Model / method' : 'Method'}</th>${groupCells}</tr><tr>${secondRow}</tr></thead>`;
  }

  function rowMarkup(row, columns, index) {
    const isOurs = row.method.startsWith('ReVuE');
    const label = isOurs ? 'ReVuE' : row.method;
    return `<tr class="${isOurs ? 'ours' : ''}" data-row="${index}"><th scope="row">${escape(label)}</th>${columns.map((column, col) => `<td data-row="${index}" data-col="${col}" tabindex="${index === 0 && col === 0 ? 0 : -1}" class="${row.best_opd_columns.includes(column.id) ? 'is-best' : ''}" aria-label="${escape(row.family + ', ' + label + ', ' + column.category + ', ' + column.label + ': ' + row.values[column.id] + '%')}">${escape(row.values[column.id])}</td>`).join('')}</tr>`;
  }

  function renderResults() {
    const columns = columnsForView();
    const rows = content.results.rows.filter(row => row.family === state.family);
    $('#results-table').innerHTML = tableHeaders(columns) + `<tbody>${rows.map((row, index) => rowMarkup(row, columns, index)).join('')}</tbody>`;
    $('#table-caption').textContent = state.metric === 'Overview' ? 'Weighted-average accuracy (%). Higher is better.' : `${state.metric} accuracy (%). Higher is better.`;
    $('#table-focus').textContent = 'Hover or focus a value to compare.';
    $$('.family-tabs button').forEach(button => {
      const selected = button.dataset.family === state.family;
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
      if (selected) $('#results-panel').setAttribute('aria-labelledby', button.id);
    });
    $$('.metric-tabs button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.metric === state.metric)));
  }

  function renderFullResults() {
    const columns = content.results.columns;
    let rows = '', currentFamily = '';
    content.results.rows.forEach((row, index) => {
      if (row.family !== currentFamily) {
        currentFamily = row.family;
        rows += `<tr class="family-divider"><th colspan="${columns.length + 1}">${escape(currentFamily)}</th></tr>`;
      }
      rows += rowMarkup(row, columns, index);
    });
    $('#full-results-table').innerHTML = tableHeaders(columns, true) + `<tbody>${rows}</tbody>`;
  }

  function attachTableInteraction(table) {
    function reset() {
      $$('.hover-row,.hover-col,.active-cell,.active-group', table).forEach(element => element.classList.remove('hover-row','hover-col','active-cell','active-group'));
    }
    function highlight(target) {
      const cell = target.closest('td,th');
      if (!cell || !table.contains(cell)) return;
      reset();
      const row = cell.closest('tbody tr[data-row]');
      if (row) row.classList.add('hover-row');
      const indices = cell.hasAttribute('data-cols') ? cell.dataset.cols.split(',') : cell.hasAttribute('data-col') ? [cell.dataset.col] : [];
      indices.forEach(col => $$(`[data-col="${col}"]`, table).forEach(element => element.classList.add('hover-col')));
      $$('th[data-cols]', table).forEach(header => {
        if (header.dataset.cols.split(',').some(col => indices.includes(col))) header.classList.add('active-group');
      });
      cell.classList.add('active-cell');
      if (table.id === 'results-table' && cell.matches('td')) {
        const column = columnsForView()[Number(cell.dataset.col)];
        $('#table-focus').textContent = `${$('th', row).textContent} · ${column.category} ${column.id.endsWith('_avg') ? '' : ' / ' + column.label} · ${cell.textContent}%`;
      }
    }
    table.addEventListener('pointerover', event => highlight(event.target));
    table.addEventListener('pointerleave', () => {
      if (table.contains(document.activeElement)) highlight(document.activeElement);
      else reset();
      if (table.id === 'results-table' && !table.contains(document.activeElement)) $('#table-focus').textContent = 'Hover or focus a value to compare.';
    });
    table.addEventListener('focusin', event => highlight(event.target));
    table.addEventListener('focusout', event => { if (!table.contains(event.relatedTarget)) reset(); });
    table.addEventListener('click', event => { const cell = event.target.closest('td'); if (cell) cell.focus({preventScroll:true}); });
    table.addEventListener('keydown', event => {
      if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(event.key) || !event.target.matches('td')) return;
      const cell = event.target;
      let row = Number(cell.dataset.row), col = Number(cell.dataset.col);
      if (event.key === 'ArrowLeft') col--;
      if (event.key === 'ArrowRight') col++;
      if (event.key === 'ArrowUp') row--;
      if (event.key === 'ArrowDown') row++;
      if (event.key === 'Home') col = 0;
      if (event.key === 'End') col = $$('td', cell.parentElement).length - 1;
      const next = $(`td[data-row="${row}"][data-col="${col}"]`, table);
      if (next) {
        event.preventDefault();
        $$('td[tabindex="0"]', table).forEach(element => element.tabIndex = -1);
        next.tabIndex = 0;
        next.focus();
      }
    });
  }

  function renderContent() {
    const authors = content.authors || [];
    if (authors.length) {
      $('#author-list').innerHTML = authors.map(author => {
        const name = escape(author.name);
        const link = author.url && /^https?:\/\//.test(author.url) ? `<a href="${escape(author.url)}" target="_blank" rel="noopener">${name}</a>` : `<span>${name}</span>`;
        const marks = [...(author.affiliations || []), ...(author.marks || [])].join(',');
        return `<span class="author-name">${link}${marks ? `<sup>${escape(marks)}</sup>` : ''}</span>`;
      }).join('');
      $('#affiliation-list').innerHTML = (content.affiliations || []).map(item => `<span><sup>${escape(item.id)}</sup>${escape(item.name)}</span>`).join('');
      $('#author-notes').textContent = (content.author_notes || []).join(' · ');
      if (content.correspondence?.email) {
        const email = content.correspondence.email;
        $('#author-notes').insertAdjacentHTML('beforeend', `<span class="correspondence">Correspondence to <a href="mailto:${escape(email)}">${escape(email)}</a></span>`);
      }
      $('#author-notes').hidden = !(content.author_notes || []).length;
    }
    const sentences = content.abstract_sentences;
    $('#abstract-copy').innerHTML = [sentences.slice(0,4), sentences.slice(4,9), sentences.slice(9)].map(part => `<p>${escape(part.join(' '))}</p>`).join('');
    $('#tldr-list').innerHTML = content.tldr.map((item, index) => `<li data-index="0${index + 1}">${escape(item)}</li>`).join('');
    $('#method-steps').innerHTML = content.method_steps.map(step => `<article class="method-step"><span class="step-number">${escape(step.number)}</span><h3>${escape(step.title)}</h3><p>${escape(step.description)}</p></article>`).join('');
    $('#method-note').textContent = content.method_footnote;
    const overview = content.cases.find(item => item.id === 'plate-evidence-chain');
    const islands = content.cases.find(item => item.id === 'two-islands');
    $('#overview-caption').innerHTML = overview.paragraphs.map(paragraph => `<p>${escape(paragraph)}</p>`).join('');
    $('#islands-intro').textContent = islands.paragraphs[0];
    $('#islands-explanation').innerHTML = `<p>${escape(islands.paragraphs[1])}</p>`;
    $('#case-gallery').innerHTML = content.cases.filter(item => !['plate-evidence-chain','two-islands'].includes(item.id)).map(item => `<article class="case-narrative" id="case-${escape(item.id)}"><div class="narrative-copy"><h3>${escape(item.title)}</h3><p class="case-lead">${escape(item.lead)}</p><p>${escape(item.paragraphs[0])}</p></div><figure class="paper-figure"><button class="case-figure-open" data-case="${escape(item.id)}" aria-label="Enlarge case: ${escape(item.title)}"><img src="${escape(item.image)}" alt="${escape(item.short_description)}" width="${item.preview_dimensions[0]}" height="${item.preview_dimensions[1]}" loading="lazy"></button><figcaption><a href="${escape(item.image)}" target="_blank" rel="noopener">SVG</a><span>·</span><a href="${escape(item.pdf)}" target="_blank" rel="noopener">PDF</a></figcaption></figure><div class="narrative-copy"><p>${escape(item.paragraphs[1])}</p></div></article>`).join('');
    $('#main-results-intro').textContent = content.findings[0].body;
    $('#findings-flow').innerHTML = content.findings.slice(1).map(finding => {
      const figure = finding.id === 'accuracy-and-efficiency' ? 'efficiency' : finding.id === 'sparse-impact' ? 'impact-sparsity' : null;
      const figureHTML = figure ? `<figure class="finding-figure ${figure === 'impact-sparsity' ? 'compact-figure' : ''}"><img src="assets/figures/${figure}.svg" alt="${escape(finding.title)}" loading="lazy"><figcaption><a href="assets/figures/${figure}.svg" target="_blank" rel="noopener">SVG</a><span>·</span><a href="assets/figures/${figure}.pdf" target="_blank" rel="noopener">PDF</a></figcaption></figure>` : '';
      const table = finding.table;
      const tableHTML = table ? `<div class="table-scroll finding-table" tabindex="0" role="region" aria-label="${escape(finding.title)}"><table id="finding-${escape(finding.id)}"><caption>${escape(finding.title)} — Qwen2.5-VL-7B</caption><thead><tr>${table.columns.map((label,index) => `<th scope="col" ${index ? `data-col="${index-1}"` : ''}>${escape(label)}${index ? ' ↑' : ''}</th>`).join('')}</tr></thead><tbody>${table.rows.map((row,index) => `<tr data-row="${index}" class="${row.method.includes('ReVuE') || row.method.includes('reweighting') ? 'ours' : ''}"><th scope="row">${escape(row.method)}</th>${row.values.map((value,col) => `<td data-row="${index}" data-col="${col}" tabindex="${index === 0 && col === 0 ? 0 : -1}" class="${table.best_by_column[col].includes(row.method) ? 'is-best' : ''}">${escape(value)}</td>`).join('')}</tr>`).join('')}</tbody></table></div><p class="figure-note">Qwen2.5-VL-7B · Weighted-average accuracy (%). Higher is better.</p>` : '';
      return `<section class="section finding-section" id="${escape(finding.id)}"><div class="narrative-copy"><h2>${escape(finding.title)}</h2><p>${escape(finding.sentences.slice(0,2).join(' '))}</p></div>${figureHTML}${tableHTML}<div class="narrative-copy finding-takeaway"><p>${escape(finding.sentences[2])}</p></div></section>`;
    }).join('');
    $('#citation-code').textContent = content.citation.bibtex;
    renderResults();
    renderFullResults();
    attachTableInteraction($('#results-table'));
    attachTableInteraction($('#full-results-table'));
    $$('.finding-table table').forEach(attachTableInteraction);
    startAnimations();
  }

  async function startAnimations() {
    try {
      const response = await fetch('assets/animations/metadata.json');
      if (!response.ok) return;
      const animations = await response.json();
      for (const name of ['overview','islands']) {
        const holder = $(`#${name}-animation`);
        const svgResponse = await fetch(`assets/animations/${name}-staged.svg`);
        if (!svgResponse.ok) continue;
        holder.innerHTML = await svgResponse.text();
        const stageGroups = $$('[data-stage]', holder);
        if (!stageGroups.length) continue;
        const controls = $(`[data-controls="${name}"]`);
        const sequence = animations[name].stages;
        if (animations[name].tokenStreaming) $('.animation-stage', controls).setAttribute('aria-live', 'off');
        const labels = [...new Set(stageGroups.map(group => Number(group.dataset.stage)))].sort((a,b) => a-b);
        const max = Math.max(...labels);
        let stage = 1, paused = matchMedia('(prefers-reduced-motion: reduce)').matches, inView = false, timer = null;
        function paint() {
          stageGroups.forEach(group => {
            const opacity = Number(group.dataset.stage) <= stage ? '1' : '0';
            if (group.style.opacity !== opacity) group.style.opacity = opacity;
          });
          const current = stageGroups.find(group => Number(group.dataset.stage) === stage);
          $('.animation-stage', controls).textContent = stage >= max ? 'Complete view' : current?.dataset.label || 'Overview';
          const toggle = $('[data-action="toggle"]', controls);
          const action = paused ? 'Play' : 'Pause';
          if (toggle.textContent !== action) toggle.textContent = action;
          if (toggle.getAttribute('aria-label') !== `${action} ${name} animation`) toggle.setAttribute('aria-label', `${action} ${name} animation`);
        }
        function schedule() {
          clearTimeout(timer);
          if (paused || !inView || document.hidden) return;
          const duration = sequence.find(item => item.stage === stage)?.durationMs || 2000;
          timer = setTimeout(() => { stage = stage >= max ? 1 : stage+1; paint(); schedule(); }, duration);
        }
        controls.addEventListener('click', event => {
          const action = event.target.closest('[data-action]')?.dataset.action;
          if (action === 'toggle') paused = !paused;
          if (action === 'replay') { stage = 1; paused = false; }
          if (action === 'all') { stage = max; paused = true; }
          paint(); schedule();
        });
        new IntersectionObserver(entries => { inView = entries[0].isIntersecting; paint(); schedule(); }, {threshold:.15}).observe(holder);
        document.addEventListener('visibilitychange', schedule);
        stage = paused ? max : 1;
        paint();
      }
    } catch { /* Full static SVGs remain available if animation loading fails. */ }
  }

  function openCase(id) {
    const item = content?.cases.find(entry => entry.id === id);
    if (!item) return;
    $('#figure-dialog-title').textContent = item.title;
    $('#figure-dialog-category').textContent = item.category;
    $('#figure-dialog-image').src = item.image;
    $('#figure-dialog-image').alt = item.short_description;
    $('#figure-dialog-caption').textContent = item.caption;
    $('#figure-dialog-pdf').href = item.pdf;
    $('#figure-dialog').showModal();
    $('.dialog-scroll').scrollTop = 0;
    document.body.style.overflow = 'hidden';
  }

  document.addEventListener('click', event => {
    const figure = event.target.closest('[data-case]');
    if (figure) openCase(figure.dataset.case);
    const family = event.target.closest('[data-family]');
    if (family && content) { state.family = family.dataset.family; renderResults(); }
    const metric = event.target.closest('[data-metric]');
    if (metric && content) { state.metric = metric.dataset.metric; renderResults(); }
    const close = event.target.closest('.dialog-close,.resource-dismiss');
    if (close) close.closest('dialog').close();
    const resource = event.target.closest('[data-resource]');
    if (resource) {
      const isData = resource.dataset.resource === 'data';
      $('#resource-title').textContent = isData ? 'Data' : 'Checkpoints';
      $('#resource-message').textContent = isData ? 'The public data link has not been added yet. The paper describes the training data and evaluation benchmarks.' : 'The model checkpoint link has not been added yet. Training and evaluation instructions are included in the code package.';
      $('#resource-dialog').showModal();
      document.body.style.overflow = 'hidden';
    }
  });

  $$('.family-tabs [role="tab"]').forEach((button, index, buttons) => button.addEventListener('keydown', event => {
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next].click(); buttons[next].focus();
  }));

  $$('dialog').forEach(dialog => {
    dialog.addEventListener('close', () => { document.body.style.overflow = ''; });
    dialog.addEventListener('click', event => { if (event.target === dialog) { const bounds = dialog.getBoundingClientRect(); if(event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close(); } });
  });

  $('#copy-citation').addEventListener('click', async () => {
    const value = $('#citation-code').textContent;
    if (!value) return;
    try {
      await Promise.race([
        navigator.clipboard.writeText(value),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Clipboard unavailable')), 1500))
      ]);
      $('#copy-status').textContent = 'Citation copied.';
      $('#copy-citation span').textContent = 'Copied';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents($('#citation-code'));
      selection.removeAllRanges(); selection.addRange(range);
      $('#copy-status').textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
    }
  });

  fetch('content.json?v=20260930-arxiv').then(response => { if (!response.ok) throw new Error('Content unavailable'); return response.json(); }).then(data => { content = data; renderContent(); }).catch(() => { $('#abstract-copy').innerHTML = '<p>The page content could not be loaded. Please refresh the page or open the paper above.</p>'; });
})();
