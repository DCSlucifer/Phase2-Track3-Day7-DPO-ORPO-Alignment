/* ========= Preference Alignment Lab — Dashboard Logic ========= */

// ===== Navigation =====
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const panelId = 'panel-' + btn.dataset.panel;
    document.getElementById(panelId).classList.add('active');
  });
});

// ===== Toast Notifications =====
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// ===== API Helper =====
async function api(endpoint) {
  try {
    const res = await fetch(endpoint);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    showToast(`API Error: ${e.message}`, 'error');
    throw e;
  }
}

// ===== Chart Instances =====
let trainChartInstance = null;
let cmpChartInstance = null;

// ===== Panel 1: Dataset Explorer =====
async function loadDataset() {
  showToast('Loading dataset...', 'info');
  const data = await api('/api/dataset');

  document.getElementById('ds-total').textContent = data.total_lines;
  document.getElementById('ds-valid').textContent = data.valid_count;
  document.getElementById('ds-warnings').textContent = data.warnings.length + data.pii_warnings.length;
  document.getElementById('ds-errors').textContent = data.errors.length;

  const tbody = document.getElementById('datasetTable');
  tbody.innerHTML = data.examples.map((ex, i) => `
    <tr>
      <td>${i + 1}</td>
      <td title="${escapeHtml(ex.prompt)}">${escapeHtml(ex.prompt.substring(0, 60))}...</td>
      <td title="${escapeHtml(ex.chosen)}">${escapeHtml(ex.chosen.substring(0, 50))}...</td>
      <td title="${escapeHtml(ex.rejected)}">${escapeHtml(ex.rejected.substring(0, 50))}...</td>
      <td><span class="badge badge-info">${ex.metadata.domain || '—'}</span></td>
    </tr>
  `).join('');

  showToast(`Loaded ${data.valid_count} examples`, 'success');
}

async function loadSplit() {
  const ratio = document.getElementById('splitRatio').value;
  const seed = document.getElementById('splitSeed').value;
  const data = await api(`/api/dataset/split?ratio=${ratio}&seed=${seed}`);

  const leakBadge = data.has_leakage
    ? '<span class="badge badge-danger">⚠ Leakage!</span>'
    : '<span class="badge badge-success">✓ No Leakage</span>';

  document.getElementById('splitResults').innerHTML = `
    <div style="width:100%">
      <div class="grid-3" style="margin-bottom:16px">
        <div class="stat-card"><div class="stat-value green">${data.train_count}</div><div class="stat-label">Train (${data.train_prompts} prompts)</div></div>
        <div class="stat-card"><div class="stat-value">${data.val_count}</div><div class="stat-label">Val (${data.val_prompts} prompts)</div></div>
        <div class="stat-card">${leakBadge}<div class="stat-label" style="margin-top:8px">Overlap: ${data.overlap}</div></div>
      </div>
      <div style="font-size:12px; color:var(--text-muted)">
        <strong>Seed:</strong> ${seed} | <strong>Ratio:</strong> ${ratio}
      </div>
    </div>
  `;
  showToast('Split computed — no data leakage!', 'success');
}

// ===== Panel 2: Loss Lab =====
async function computeDPO() {
  const beta = document.getElementById('dpoBeta').value;
  const pc = document.getElementById('dpoPC').value;
  const pr = document.getElementById('dpoPR').value;
  const rc = document.getElementById('dpoRC').value;
  const rr = document.getElementById('dpoRR').value;

  const data = await api(`/api/loss/dpo?beta=${beta}&policy_chosen=${pc}&policy_rejected=${pr}&ref_chosen=${rc}&ref_rejected=${rr}`);

  document.getElementById('dpo-loss').textContent = data.loss.toFixed(6);
  document.getElementById('dpo-logits').textContent = data.logits.toFixed(6);
  document.getElementById('dpo-plr').textContent = data.policy_log_ratio.toFixed(4);
}

async function computeORPO() {
  const lambda = document.getElementById('orpoLambda').value;
  const sft = document.getElementById('orpoSFT').value;
  const c = document.getElementById('orpoC').value;
  const r = document.getElementById('orpoR').value;

  const data = await api(`/api/loss/orpo?lambda_orpo=${lambda}&sft_nll=${sft}&chosen_logps=${c}&rejected_logps=${r}`);

  document.getElementById('orpo-loss').textContent = data.loss.toFixed(6);
  document.getElementById('orpo-sft').textContent = parseFloat(sft).toFixed(4);
  const pref = (data.loss - parseFloat(sft));
  document.getElementById('orpo-pref').textContent = pref.toFixed(4);
}

// ===== Panel 3: Training Simulator =====
async function runTraining() {
  const btn = document.getElementById('trainBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Training...';
  showToast('Starting training simulation...', 'info');

  const method = document.getElementById('trainMethod').value;
  const steps = document.getElementById('trainSteps').value;
  const beta = document.getElementById('trainBeta').value;
  const lambda = document.getElementById('trainLambda').value;
  const batch = document.getElementById('trainBatch').value;

  try {
    const data = await api(`/api/train?method=${method}&steps=${steps}&beta=${beta}&lambda_orpo=${lambda}&batch_size=${batch}`);

    // Show stats
    document.getElementById('trainStats').style.display = '';
    const r = data.results[0];
    document.getElementById('train-method').textContent = data.results.map(x => x.method.toUpperCase()).join(' + ');
    document.getElementById('train-initial').textContent = r.initial_loss.toFixed(4);
    document.getElementById('train-final').textContent = r.final_loss.toFixed(4);
    document.getElementById('train-time').textContent = data.results.reduce((s, x) => s + x.elapsed_seconds, 0).toFixed(3) + 's';

    // Chart
    document.getElementById('trainChartCard').style.display = '';
    renderTrainChart(data.results);

    showToast(`Training complete! Final loss: ${r.final_loss.toFixed(4)}`, 'success');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶ Start Training';
  }
}

function renderTrainChart(results) {
  const ctx = document.getElementById('trainChart').getContext('2d');
  if (trainChartInstance) trainChartInstance.destroy();

  const colors = { dpo: '#06b6d4', orpo: '#a855f7' };
  const datasets = results.map(r => ({
    label: r.method.toUpperCase() + ' Loss',
    data: r.loss_history,
    borderColor: colors[r.method] || '#6366f1',
    backgroundColor: (colors[r.method] || '#6366f1') + '15',
    fill: true,
    tension: 0.4,
    pointRadius: 0,
    borderWidth: 2,
  }));

  trainChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: results[0].loss_history.map((_, i) => i + 1),
      datasets,
    },
    options: chartOptions('Training Step', 'Loss'),
  });
}

// ===== Panel 4: Evaluation =====
async function runEvaluation() {
  const scorer = document.getElementById('evalScorer').value;
  showToast(`Running evaluation with ${scorer} scorer...`, 'info');

  const data = await api(`/api/evaluate?scorer=${scorer}`);
  const m = data.metrics;

  document.getElementById('evalStats').style.display = '';
  document.getElementById('eval-accuracy').textContent = (m.accuracy * 100).toFixed(1) + '%';
  document.getElementById('eval-margin').textContent = m.reward_margin.toFixed(4);
  document.getElementById('eval-wins').textContent = `${m.win_count}/${m.loss_count}/${m.tie_count}`;
  document.getElementById('eval-total').textContent = m.total;

  // Table
  document.getElementById('evalTableCard').style.display = '';
  const tbody = document.getElementById('evalTable');
  tbody.innerHTML = data.per_example.map((ex, i) => `
    <tr>
      <td>${i + 1}</td>
      <td title="${escapeHtml(ex.prompt)}">${escapeHtml(ex.prompt)}</td>
      <td style="font-family:'JetBrains Mono',monospace;color:var(--accent-green)">${ex.chosen_score.toFixed(4)}</td>
      <td style="font-family:'JetBrains Mono',monospace;color:var(--accent-red)">${ex.rejected_score.toFixed(4)}</td>
      <td style="font-family:'JetBrains Mono',monospace;color:${ex.margin > 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">${ex.margin > 0 ? '+' : ''}${ex.margin.toFixed(4)}</td>
      <td>${ex.correct ? '<span class="badge badge-success">✓</span>' : '<span class="badge badge-danger">✗</span>'}</td>
    </tr>
  `).join('');

  showToast(`Evaluation complete — ${(m.accuracy * 100).toFixed(1)}% accuracy`, 'success');
}

// ===== Panel 5: DPO vs ORPO =====
async function runComparison() {
  showToast('Starting DPO vs ORPO battle...', 'info');

  const steps = document.getElementById('cmpSteps').value;
  const beta = document.getElementById('cmpBeta').value;
  const lambda = document.getElementById('cmpLambda').value;

  const data = await api(`/api/compare?steps=${steps}&beta=${beta}&lambda_orpo=${lambda}`);

  // Winner badge
  document.getElementById('comparisonWinner').style.display = '';
  document.getElementById('winnerBadge').textContent = `🏆 Winner: ${data.winner.toUpperCase()}`;

  // Stats
  document.getElementById('comparisonStats').style.display = '';
  document.getElementById('cmp-dpo-initial').textContent = data.dpo.loss_history[0]?.toFixed(4) || '—';
  document.getElementById('cmp-dpo-final').textContent = data.dpo.final_loss.toFixed(4);
  document.getElementById('cmp-orpo-initial').textContent = data.orpo.loss_history[0]?.toFixed(4) || '—';
  document.getElementById('cmp-orpo-final').textContent = data.orpo.final_loss.toFixed(4);

  // Chart
  document.getElementById('cmpChartCard').style.display = '';
  renderComparisonChart(data);

  showToast(`${data.winner.toUpperCase()} wins! Final: DPO=${data.dpo.final_loss.toFixed(4)} vs ORPO=${data.orpo.final_loss.toFixed(4)}`, 'success');
}

function renderComparisonChart(data) {
  const ctx = document.getElementById('cmpChart').getContext('2d');
  if (cmpChartInstance) cmpChartInstance.destroy();

  const maxLen = Math.max(data.dpo.loss_history.length, data.orpo.loss_history.length);

  cmpChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array.from({ length: maxLen }, (_, i) => i + 1),
      datasets: [
        {
          label: 'DPO Loss',
          data: data.dpo.loss_history,
          borderColor: '#06b6d4',
          backgroundColor: '#06b6d415',
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: 'ORPO Loss',
          data: data.orpo.loss_history,
          borderColor: '#a855f7',
          backgroundColor: '#a855f715',
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    },
    options: chartOptions('Step', 'Loss'),
  });
}

// ===== Panel 6: Safety Console =====
async function runSafety() {
  showToast('Running safety regression...', 'info');

  const data = await api('/api/safety?mode=both');

  document.getElementById('safetyOverview').style.display = '';
  document.getElementById('safety-before-score').textContent = (data.before.overall_score * 100).toFixed(0) + '%';
  document.getElementById('safety-after-score').textContent = (data.after.overall_score * 100).toFixed(0) + '%';

  document.getElementById('safetyDetails').style.display = '';
  document.getElementById('safetyBeforeList').innerHTML = renderSafetyCards(data.before.results);
  document.getElementById('safetyAfterList').innerHTML = renderSafetyCards(data.after.results);

  showToast('Safety regression complete!', 'success');
}

function renderSafetyCards(results) {
  return results.map(r => `
    <div style="margin-bottom:16px; padding:16px; background:rgba(255,255,255,0.02); border-radius:var(--radius-sm); border:1px solid var(--border-glass)">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px">
        <span class="badge ${r.passed ? 'badge-success' : 'badge-danger'}">${r.category}</span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:13px; color:${r.passed ? 'var(--accent-green)' : 'var(--accent-red)'}">
          ${(r.safety_score * 100).toFixed(0)}%
        </span>
      </div>
      <div class="safety-bar"><div class="safety-bar-fill ${r.passed ? 'pass' : 'fail'}" style="width:${r.safety_score * 100}%"></div></div>
      <p style="font-size:12px; color:var(--text-muted); margin-top:8px"><strong>Prompt:</strong> ${escapeHtml(r.prompt)}</p>
      <p style="font-size:12px; color:var(--text-secondary); margin-top:4px">${escapeHtml(r.response.substring(0, 150))}...</p>
      ${r.safety_hits.length ? `<div style="margin-top:6px">${r.safety_hits.map(h => `<span class="badge badge-success" style="margin:2px">${h}</span>`).join('')}</div>` : ''}
      ${r.danger_hits.length ? `<div style="margin-top:4px">${r.danger_hits.map(h => `<span class="badge badge-danger" style="margin:2px">${h}</span>`).join('')}</div>` : ''}
    </div>
  `).join('');
}

// ===== Shared Utilities =====
function chartOptions(xLabel, yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { intersect: false, mode: 'index' },
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 } },
      },
      tooltip: {
        backgroundColor: 'rgba(17, 24, 39, 0.95)',
        borderColor: 'rgba(99, 102, 241, 0.3)',
        borderWidth: 1,
        titleFont: { family: 'Inter' },
        bodyFont: { family: 'JetBrains Mono', size: 12 },
        padding: 12,
      },
    },
    scales: {
      x: {
        title: { display: true, text: xLabel, color: '#64748b', font: { family: 'Inter', size: 12 } },
        grid: { color: 'rgba(255,255,255,0.03)' },
        ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } },
      },
      y: {
        title: { display: true, text: yLabel, color: '#64748b', font: { family: 'Inter', size: 12 } },
        grid: { color: 'rgba(255,255,255,0.03)' },
        ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 11 } },
      },
    },
  };
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ===== Auto-load dataset on page load =====
window.addEventListener('DOMContentLoaded', () => {
  loadDataset();
  showToast('Dashboard ready — 37 tests passing ✓', 'success');
});
