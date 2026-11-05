// Global State
let globalMetadata = null;
let globalEda = null;
let globalDict = null;
let consensusChartInstance = null;
let currentTablePage = 1;
let currentTableSort = 'id';
let currentTableOrder = 'asc';

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  loadInitialData();
});

// TAB SWITCHING
function initTabs() {
  const buttons = document.querySelectorAll('.nav-btn');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      buttons.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add('active');

      // Refresh charts if opening EDA or Benchmark
      if (targetId === 'tab-eda' && !window.edaChartsRendered) {
        renderEdaCharts();
      }
      if (targetId === 'tab-models' && !window.modelChartsRendered) {
        renderModelCharts();
      }
    });
  });
}

// FETCH INITIAL DATA
async function loadInitialData() {
  try {
    const [metaRes, edaRes, dictRes] = await Promise.all([
      fetch('/api/metadata').then(r => r.json()),
      fetch('/api/eda').then(r => r.json()),
      fetch('/api/dictionary').then(r => r.json())
    ]);

    globalMetadata = metaRes;
    globalEda = edaRes;
    globalDict = dictRes;

    // Populate Neighborhood Dropdown in Table Filter
    populateNeighborhoodFilter();

    // Initial table load
    loadTestTable();

    // Populate Leaderboard
    populateLeaderboard();

    // Populate Dictionary
    populateDictionary();

    // Trigger initial prediction for default form
    triggerPredict();

  } catch (err) {
    console.error('Failed to load initial data:', err);
  }
}

// PRESET HOUSE PROFILES
const presets = {
  luxury: {
    Neighborhood: 'NridgHt',
    OverallQual: 9,
    OverallCond: 5,
    YearBuilt: 2008,
    YearRemodAdd: 2009,
    GrLivArea: 2800,
    TotalBsmtSF: 1400,
    firstFlrSF: 1400,
    secondFlrSF: 1400,
    LotArea: 14500,
    FullBath: 3,
    HalfBath: 1,
    BedroomAbvGr: 4,
    TotRmsAbvGrd: 9,
    KitchenQual: 'Ex',
    Fireplaces: 2,
    GarageCars: 3,
    GarageArea: 820,
    CentralAir: 'Y',
    WoodDeckSF: 240
  },
  suburban: {
    Neighborhood: 'CollgCr',
    OverallQual: 7,
    OverallCond: 5,
    YearBuilt: 2003,
    YearRemodAdd: 2004,
    GrLivArea: 1850,
    TotalBsmtSF: 950,
    firstFlrSF: 1050,
    secondFlrSF: 800,
    LotArea: 9600,
    FullBath: 2,
    HalfBath: 1,
    BedroomAbvGr: 3,
    TotRmsAbvGrd: 7,
    KitchenQual: 'Gd',
    Fireplaces: 1,
    GarageCars: 2,
    GarageArea: 500,
    CentralAir: 'Y',
    WoodDeckSF: 140
  },
  craftsman: {
    Neighborhood: 'OldTown',
    OverallQual: 6,
    OverallCond: 7,
    YearBuilt: 1938,
    YearRemodAdd: 1995,
    GrLivArea: 1450,
    TotalBsmtSF: 850,
    firstFlrSF: 900,
    secondFlrSF: 550,
    LotArea: 6500,
    FullBath: 1,
    HalfBath: 1,
    BedroomAbvGr: 3,
    TotRmsAbvGrd: 6,
    KitchenQual: 'TA',
    Fireplaces: 1,
    GarageCars: 1,
    GarageArea: 320,
    CentralAir: 'Y',
    WoodDeckSF: 60
  },
  starter: {
    Neighborhood: 'Edwards',
    OverallQual: 5,
    OverallCond: 6,
    YearBuilt: 1965,
    YearRemodAdd: 1970,
    GrLivArea: 1050,
    TotalBsmtSF: 900,
    firstFlrSF: 1050,
    secondFlrSF: 0,
    LotArea: 7200,
    FullBath: 1,
    HalfBath: 0,
    BedroomAbvGr: 3,
    TotRmsAbvGrd: 5,
    KitchenQual: 'TA',
    Fireplaces: 0,
    GarageCars: 1,
    GarageArea: 280,
    CentralAir: 'Y',
    WoodDeckSF: 0
  }
};

function loadPreset(key) {
  const data = presets[key];
  if (!data) return;

  for (const [k, v] of Object.entries(data)) {
    const el = document.getElementById(k);
    if (el) {
      el.value = v;
      if (k === 'OverallQual') {
        const badge = document.getElementById('qual-val');
        if (badge) badge.innerText = v;
      }
    }
  }
  triggerPredict();
}

function resetForm() {
  loadPreset('suburban');
}

function syncFloorAreas() {
  const grLiv = parseFloat(document.getElementById('GrLivArea').value) || 1500;
  const first = parseFloat(document.getElementById('firstFlrSF').value) || 0;
  const second = parseFloat(document.getElementById('secondFlrSF').value) || 0;

  if (first === 0 && second === 0) {
    document.getElementById('firstFlrSF').value = Math.round(grLiv * 0.6);
    document.getElementById('secondFlrSF').value = Math.round(grLiv * 0.4);
  }
}

function syncGarageArea() {
  const cars = parseInt(document.getElementById('GarageCars').value) || 0;
  const areaInput = document.getElementById('GarageArea');
  if (cars === 0) {
    areaInput.value = 0;
  } else if (cars === 1) {
    areaInput.value = 300;
  } else if (cars === 2) {
    areaInput.value = 500;
  } else if (cars === 3) {
    areaInput.value = 750;
  } else {
    areaInput.value = 950;
  }
}

// VALUATION PREDICTION
async function handlePredict(e) {
  if (e) e.preventDefault();
  triggerPredict();
}

async function triggerPredict() {
  const form = document.getElementById('prediction-form');
  if (!form) return;

  const btn = document.getElementById('predict-btn');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Calculating Valuation...';
  }

  const formData = new FormData(form);
  const payload = {};
  formData.forEach((val, key) => {
    payload[key] = isNaN(val) || val === '' ? val : parseFloat(val);
  });

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error('Prediction API call failed');
    const result = await res.json();

    updateValuationDisplay(result);
  } catch (err) {
    console.error('Prediction failed:', err);
    alert('Failed to calculate valuation: ' + err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<i class="fa-solid fa-sparkles"></i> Calculate Valuation';
    }
  }
}

function updateValuationDisplay(res) {
  document.getElementById('res-price').innerText = Math.round(res.predicted_price).toLocaleString();
  document.getElementById('res-range').innerText = `$${Math.round(res.low_price).toLocaleString()} — $${Math.round(res.high_price).toLocaleString()}`;
  document.getElementById('res-ppsqft').innerText = `$${res.price_per_sqft.toFixed(2)}`;
  document.getElementById('res-totalsf').innerText = `${Math.round(res.inputs_summary.total_sqft).toLocaleString()} SF`;
  document.getElementById('res-qual').innerText = `${res.inputs_summary.overall_quality} / 10`;

  renderConsensusChart(res.model_breakdown, res.predicted_price);
}

function renderConsensusChart(models, ensemblePrice) {
  const ctx = document.getElementById('modelConsensusChart');
  if (!ctx) return;

  const labels = Object.keys(models);
  const data = Object.values(models);

  if (consensusChartInstance) {
    consensusChartInstance.destroy();
  }

  consensusChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Predicted Valuation ($)',
        data: data,
        backgroundColor: labels.map(l => l === 'Ensemble Blend' ? '#10b981' : '#6366f1'),
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `$${Math.round(ctx.parsed.y).toLocaleString()}`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { size: 10 } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: {
            color: '#94a3b8',
            callback: (v) => `$${(v/1000).toFixed(0)}k`,
            font: { size: 10 }
          }
        }
      }
    }
  });
}

// EDA CHARTS
window.edaChartsRendered = false;
function renderEdaCharts() {
  if (!globalEda) return;
  window.edaChartsRendered = true;

  // 1. Price Distribution Chart
  const dist = globalEda.price_distribution;
  if (dist) {
    const ctx = document.getElementById('priceDistChart');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: dist.bin_centers.map(c => `$${Math.round(c/1000)}k`),
        datasets: [{
          label: 'Properties Count',
          data: dist.counts,
          backgroundColor: '#6366f1',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => dist.bins[items[0].dataIndex],
              label: (item) => `${item.parsed.y} Homes`
            }
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#94a3b8', maxTicksLimit: 12 } },
          y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
        }
      }
    });
  }

  // 2. Neighborhood Chart (Top 12)
  const neighs = globalEda.neighborhood_stats.slice(0, 12);
  if (neighs) {
    const ctx = document.getElementById('neighborhoodChart');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: neighs.map(n => n.neighborhood),
        datasets: [{
          label: 'Median Sale Price ($)',
          data: neighs.map(n => n.median),
          backgroundColor: '#06b6d4',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: { label: (ctx) => `Median: $${Math.round(ctx.parsed.x).toLocaleString()}` }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#94a3b8', callback: (v) => `$${(v/1000).toFixed(0)}k` }
          },
          y: { grid: { display: false }, ticks: { color: '#94a3b8' } }
        }
      }
    });
  }

  // 3. Quality vs Price Chart
  const qual = globalEda.qual_stats;
  if (qual) {
    const ctx = document.getElementById('qualityPriceChart');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: qual.map(q => `Qual ${q.quality}`),
        datasets: [{
          label: 'Median Sale Price',
          data: qual.map(q => q.median),
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.15)',
          fill: true,
          tension: 0.35,
          pointRadius: 5,
          pointBackgroundColor: '#10b981'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: { label: (ctx) => `Median: $${Math.round(ctx.parsed.y).toLocaleString()}` }
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
          y: {
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#94a3b8', callback: (v) => `$${(v/1000).toFixed(0)}k` }
          }
        }
      }
    });
  }

  // 4. Living Area Scatter Chart
  const scatterPts = globalEda.scatter_living_area;
  if (scatterPts) {
    const ctx = document.getElementById('livingAreaScatterChart');
    new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Ames Houses',
          data: scatterPts.map(p => ({ x: p.x, y: p.y })),
          backgroundColor: 'rgba(129, 140, 248, 0.65)',
          pointRadius: 4,
          pointHoverRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.parsed.x} SF → $${Math.round(ctx.parsed.y).toLocaleString()}`
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#94a3b8' },
            title: { display: true, text: 'Above Ground Living Area (Sq.Ft)', color: '#64748b' }
          },
          y: {
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#94a3b8', callback: (v) => `$${(v/1000).toFixed(0)}k` },
            title: { display: true, text: 'Sale Price ($)', color: '#64748b' }
          }
        }
      }
    });
  }

  // 5. Decade Chart
  const decades = globalEda.decade_stats;
  if (decades) {
    const ctx = document.getElementById('decadeChart');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: decades.map(d => d.decade),
        datasets: [{
          label: 'Average Price',
          data: decades.map(d => d.mean),
          backgroundColor: '#f59e0b',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: { label: (ctx) => `Avg Price: $${Math.round(ctx.parsed.y).toLocaleString()}` }
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
          y: {
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#94a3b8', callback: (v) => `$${(v/1000).toFixed(0)}k` }
          }
        }
      }
    });
  }

  // 6. Correlations Chart (Top 10)
  const corrs = globalEda.top_correlations.slice(0, 10);
  if (corrs) {
    const ctx = document.getElementById('correlationChart');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: corrs.map(c => c.feature),
        datasets: [{
          label: 'Correlation with Price (r)',
          data: corrs.map(c => c.correlation),
          backgroundColor: '#818cf8',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: { label: (ctx) => `Pearson r: ${ctx.parsed.x.toFixed(3)}` }
          }
        },
        scales: {
          x: {
            min: 0,
            max: 1.0,
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#94a3b8' }
          },
          y: { grid: { display: false }, ticks: { color: '#94a3b8' } }
        }
      }
    });
  }
}

// TEST TABLE
function populateNeighborhoodFilter() {
  if (!globalEda || !globalEda.neighborhood_stats) return;
  const select = document.getElementById('filter-neighborhood');
  if (!select) return;

  globalEda.neighborhood_stats.forEach(n => {
    const opt = document.createElement('option');
    opt.value = n.neighborhood;
    opt.textContent = `${n.neighborhood} (${n.count})`;
    select.appendChild(opt);
  });
}

async function loadTestTable() {
  const tbody = document.getElementById('test-table-body');
  if (!tbody) return;

  const search = document.getElementById('table-search')?.value || '';
  const neighborhood = document.getElementById('filter-neighborhood')?.value || 'all';

  let url = `/api/test-data?page=${currentTablePage}&limit=25&sort_by=${currentTableSort}&order=${currentTableOrder}`;
  if (search) url += `&search=${encodeURIComponent(search)}`;
  if (neighborhood && neighborhood !== 'all') url += `&neighborhood=${encodeURIComponent(neighborhood)}`;

  try {
    const res = await fetch(url);
    const data = await res.json();

    tbody.innerHTML = '';
    if (data.items.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; padding: 2rem;">No matching properties found.</td></tr>';
      return;
    }

    data.items.forEach(item => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>#${item.id}</strong></td>
        <td><span class="badge badge-secondary">${item.neighborhood}</span></td>
        <td>${item.overallQual} / 10</td>
        <td>${item.yearBuilt}</td>
        <td>${item.grLivArea.toLocaleString()} SF</td>
        <td>${item.bedroomAbvGr} bd / ${item.fullBath} ba</td>
        <td>${item.garageCars} car</td>
        <td class="price-cell">$${Math.round(item.predictedPrice).toLocaleString()}</td>
        <td>
          <button class="btn btn-sm btn-outline" onclick="loadHouseIntoEstimator(${JSON.stringify(item).replace(/"/g, '&quot;')})">
            <i class="fa-solid fa-arrow-up-right-from-square"></i> Test
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    // Update pagination
    const pageInfo = document.getElementById('page-info');
    if (pageInfo) {
      const start = (data.page - 1) * data.limit + 1;
      const end = Math.min(start + data.limit - 1, data.total);
      pageInfo.innerText = `Showing ${start} to ${end} of ${data.total} records`;
    }

    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    if (prevBtn) prevBtn.disabled = data.page <= 1;
    if (nextBtn) nextBtn.disabled = data.page >= data.pages;

  } catch (err) {
    console.error('Failed to load test table:', err);
  }
}

function handleTableSearch() {
  currentTablePage = 1;
  loadTestTable();
}

function handleTableFilter() {
  currentTablePage = 1;
  loadTestTable();
}

function sortTable(column) {
  if (currentTableSort === column) {
    currentTableOrder = currentTableOrder === 'asc' ? 'desc' : 'asc';
  } else {
    currentTableSort = column;
    currentTableOrder = 'asc';
  }
  loadTestTable();
}

function changePage(delta) {
  currentTablePage += delta;
  if (currentTablePage < 1) currentTablePage = 1;
  loadTestTable();
}

function loadHouseIntoEstimator(house) {
  // Populate form with this test house
  document.getElementById('Neighborhood').value = house.neighborhood;
  document.getElementById('OverallQual').value = house.overallQual;
  document.getElementById('qual-val').innerText = house.overallQual;
  document.getElementById('YearBuilt').value = house.yearBuilt;
  document.getElementById('YearRemodAdd').value = house.yearBuilt;
  document.getElementById('GrLivArea').value = house.grLivArea;
  document.getElementById('TotalBsmtSF').value = house.totalBsmtSF || Math.round(house.grLivArea * 0.6);
  document.getElementById('BedroomAbvGr').value = house.bedroomAbvGr;
  document.getElementById('FullBath').value = house.fullBath;
  document.getElementById('GarageCars').value = house.garageCars;

  syncFloorAreas();
  syncGarageArea();

  // Switch to predictor tab
  document.querySelector('[data-tab="tab-predictor"]').click();
  triggerPredict();
}

// BENCHMARKS TAB
window.modelChartsRendered = false;
function populateLeaderboard() {
  if (!globalMetadata || !globalMetadata.results) return;
  const tbody = document.getElementById('leaderboard-body');
  if (!tbody) return;

  tbody.innerHTML = '';
  globalMetadata.results.forEach(res => {
    const isWinner = res.name === 'Ensemble Blend';
    const tr = document.createElement('tr');
    if (isWinner) tr.classList.add('row-winner');

    tr.innerHTML = `
      <td>
        <strong>${res.name}</strong>
        ${isWinner ? '<span class="badge badge-success" style="margin-left: 0.5rem;"><i class="fa-solid fa-crown"></i> Champion</span>' : ''}
      </td>
      <td><strong>${res.RMSLE.toFixed(4)}</strong></td>
      <td>$${res['RMSE ($)'].toLocaleString()}</td>
      <td>$${res['MAE ($)'].toLocaleString()}</td>
      <td>${res['R2'].toFixed(4)}</td>
      <td>
        <span class="badge ${isWinner ? 'badge-success' : 'badge-primary'}">
          ${isWinner ? 'Production' : 'Evaluated'}
        </span>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function renderModelCharts() {
  if (!globalMetadata || !globalMetadata.top_features) return;
  window.modelChartsRendered = true;

  const top15 = globalMetadata.top_features.slice(0, 15);
  const ctx = document.getElementById('importanceChart');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: top15.map(f => f.feature),
      datasets: [{
        label: 'Relative Importance Score',
        data: top15.map(f => f.importance),
        backgroundColor: '#6366f1',
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Weight: ${(ctx.parsed.x * 100).toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#94a3b8', callback: (v) => `${(v * 100).toFixed(0)}%` }
        },
        y: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 11 } } }
      }
    }
  });
}

// DATA DICTIONARY
function populateDictionary(filter = '') {
  if (!globalDict) return;
  const container = document.getElementById('dict-container');
  if (!container) return;

  container.innerHTML = '';
  const entries = Object.entries(globalDict);
  let count = 0;

  entries.forEach(([name, data]) => {
    if (filter && !name.toLowerCase().includes(filter.toLowerCase()) && 
        !data.description.toLowerCase().includes(filter.toLowerCase())) {
      return;
    }
    count++;

    const card = document.createElement('div');
    card.classList.add('dict-card');

    let optionsHtml = '';
    if (data.options && data.options.length > 0) {
      optionsHtml = `
        <div class="dict-options-title">Allowed Values / Codes:</div>
        <div class="dict-options">
          ${data.options.map(opt => `<span class="dict-tag" title="${opt.meaning}">${opt.code}${opt.meaning ? ': ' + opt.meaning : ''}</span>`).join('')}
        </div>
      `;
    }

    card.innerHTML = `
      <div class="dict-name">${name}</div>
      <div class="dict-desc">${data.description || 'Property feature attribute.'}</div>
      ${optionsHtml}
    `;
    container.appendChild(card);
  });

  const counter = document.getElementById('dict-counter');
  if (counter) counter.innerText = `Showing ${count} of ${entries.length} Attributes`;
}

function handleDictSearch() {
  const query = document.getElementById('dict-search')?.value || '';
  populateDictionary(query);
}
