/**
 * Household Income Predictor — Web Application
 * ==============================================
 * Client-side prediction engine + Chart.js analytics
 * Model: Random Forest (best) / Ridge Regression (web prediction)
 */

// ============================================================
// Global State
// ============================================================
let modelParams = null;
let modelMetadata = null;
let chartData = null;
let isModelLoaded = false;

// Chart.js global config
const CHART_COLORS = {
    primary: '#6366f1',
    primaryLight: '#818cf8',
    secondary: '#06b6d4',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    purple: '#8b5cf6',
    pink: '#ec4899',
    palette: ['#6366f1','#06b6d4','#10b981','#f59e0b','#ec4899','#8b5cf6','#ef4444','#14b8a6','#f97316','#a855f7'],
    gridColor: 'rgba(148, 163, 184, 0.08)',
    textColor: '#94a3b8',
    textLight: '#64748b',
};

// ============================================================
// Initialize
// ============================================================
document.addEventListener('DOMContentLoaded', async () => {
    initNavigation();
    initFormControls();
    initScrollAnimations();
    await loadAllData();
});

async function loadAllData() {
    try {
        const [paramsRes, metaRes, chartRes] = await Promise.all([
            fetch('web_model_params.json'),
            fetch('model_metadata.json'),
            fetch('chart_data.json'),
        ]);
        modelParams = await paramsRes.json();
        modelMetadata = await metaRes.json();
        chartData = await chartRes.json();
        isModelLoaded = true;

        populateHeroStats();
        populateSummaryStats();
        renderAllCharts();
    } catch (err) {
        console.error('Data load error:', err);
        useFallbackData();
    }
}

// ============================================================
// Populate Hero & Summary Stats
// ============================================================
function populateHeroStats() {
    if (!modelMetadata) return;
    const m = modelMetadata;
    setText('statAccuracy', m.metrics?.CV_R2_Mean?.toFixed(4) || '--');
    setText('statDatapoints', (m.dataset_info?.total_rows || 10000).toLocaleString() + '+');
    setText('cardAvgIncome', '$' + fmtNum(m.income_stats?.median));
    setText('qsMedianValue', '$' + fmtNum(m.income_stats?.median));
    setText('qsMeanValue', '$' + fmtNum(m.income_stats?.mean));
}

function populateSummaryStats() {
    if (!chartData?.summary_stats) return;
    const s = chartData.summary_stats;
    setText('statTotalRecords', s.total_records.toLocaleString());
    setText('statMeanIncome', '$' + fmtNum(s.mean_income));
    setText('statMedianIncome', '$' + fmtNum(s.median_income));
    setText('statSkewness', s.skewness.toFixed(2));
}

// ============================================================
// Chart.js — Default Settings
// ============================================================
function chartDefaults() {
    Chart.defaults.color = CHART_COLORS.textColor;
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.padding = 16;
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 23, 42, 0.95)';
    Chart.defaults.plugins.tooltip.titleColor = '#f1f5f9';
    Chart.defaults.plugins.tooltip.bodyColor = '#cbd5e1';
    Chart.defaults.plugins.tooltip.borderColor = 'rgba(99,102,241,0.3)';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.padding = 12;
}

// ============================================================
// Render All Charts
// ============================================================
function renderAllCharts() {
    if (!chartData) return;
    chartDefaults();
    renderIncomeHistogram();
    renderLogIncomeHistogram();
    renderGroupedBarChart('chartEducation', chartData.income_by_education, '$');
    renderGroupedBarChart('chartOccupation', chartData.income_by_occupation, '$');
    renderGroupedBarChart('chartLocation', chartData.income_by_location, '$');
    renderGroupedBarChart('chartEmployment', chartData.income_by_employment, '$');
    renderGroupedBarChart('chartGender', chartData.income_by_gender, '$');
    renderGroupedBarChart('chartMarital', chartData.income_by_marital, '$');
    renderFeatureImportance();
    renderPercentiles();
    renderDoughnut('chartEduDist', chartData.dist_education_level);
    renderDoughnut('chartOccDist', chartData.dist_occupation);
}

// ── Income Histogram ──
function renderIncomeHistogram() {
    const d = chartData.income_histogram;
    new Chart(document.getElementById('chartIncomeHist'), {
        type: 'bar',
        data: {
            labels: d.labels,
            datasets: [{
                label: 'Số hộ gia đình',
                data: d.values,
                backgroundColor: createGradientBar('chartIncomeHist', CHART_COLORS.primary, CHART_COLORS.purple),
                borderRadius: 4,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => `Số lượng: ${ctx.raw.toLocaleString()}` } },
            },
            scales: {
                x: { grid: { display: false }, ticks: { maxTicksLimit: 8, maxRotation: 45 } },
                y: { grid: { color: CHART_COLORS.gridColor }, ticks: { callback: v => v.toLocaleString() } },
            },
        }
    });
}

// ── Log Income Histogram ──
function renderLogIncomeHistogram() {
    const d = chartData.log_income_histogram;
    new Chart(document.getElementById('chartLogIncomeHist'), {
        type: 'bar',
        data: {
            labels: d.labels,
            datasets: [{
                label: 'Số hộ gia đình',
                data: d.values,
                backgroundColor: createGradientBar('chartLogIncomeHist', CHART_COLORS.secondary, '#3b82f6'),
                borderRadius: 4,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => `Số lượng: ${ctx.raw.toLocaleString()}` } },
            },
            scales: {
                x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } },
                y: { grid: { color: CHART_COLORS.gridColor }, ticks: { callback: v => v.toLocaleString() } },
            },
        }
    });
}

// ── Grouped Bar Chart (Median + Mean) ──
function renderGroupedBarChart(canvasId, data, prefix) {
    if (!data) return;
    new Chart(document.getElementById(canvasId), {
        type: 'bar',
        data: {
            labels: data.labels,
            datasets: [
                {
                    label: 'Median',
                    data: data.median,
                    backgroundColor: CHART_COLORS.primary,
                    borderRadius: 6,
                    borderSkipped: false,
                },
                {
                    label: 'Mean',
                    data: data.mean,
                    backgroundColor: CHART_COLORS.secondary,
                    borderRadius: 6,
                    borderSkipped: false,
                },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: ${prefix}${ctx.raw.toLocaleString()}`,
                    }
                },
            },
            scales: {
                x: { grid: { display: false } },
                y: {
                    grid: { color: CHART_COLORS.gridColor },
                    ticks: { callback: v => '$' + fmtNum(v) },
                },
            },
        }
    });
}

// ── Feature Importance (Horizontal Bar) ──
function renderFeatureImportance() {
    const fi = modelMetadata?.feature_importance;
    if (!fi) return;

    const entries = Object.entries(fi).sort(([,a],[,b]) => b - a).slice(0, 12);
    const friendlyNames = {
        'Age_WorkExp_Ratio': 'Tuổi / Kinh nghiệm',
        'Work_Experience': 'Kinh nghiệm',
        'Age': 'Tuổi',
        'Dependents_per_HHSize': 'Phụ thuộc / Quy mô',
        'Occupation_encoded': 'Ngành nghề',
        'Household_Size': 'Quy mô HGĐ',
        'Employment_Status_encoded': 'Việc làm',
        'Education_Level_encoded': 'Học vấn',
        'Number_of_Dependents': 'Số phụ thuộc',
        'Type_of_Housing_encoded': 'Loại nhà ở',
        'Primary_Mode_of_Transportation_encoded': 'Phương tiện',
        'Location_encoded': 'Khu vực',
        'Marital_Status_encoded': 'Hôn nhân',
        'Homeownership_Status_encoded': 'Sở hữu nhà',
        'Gender_encoded': 'Giới tính',
    };

    new Chart(document.getElementById('chartFeatureImportance'), {
        type: 'bar',
        data: {
            labels: entries.map(([k]) => friendlyNames[k] || k),
            datasets: [{
                label: 'Importance (%)',
                data: entries.map(([,v]) => +(v * 100).toFixed(2)),
                backgroundColor: entries.map((_, i) => CHART_COLORS.palette[i % CHART_COLORS.palette.length]),
                borderRadius: 6,
                borderSkipped: false,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => `${ctx.raw.toFixed(2)}%` } },
            },
            scales: {
                x: {
                    grid: { color: CHART_COLORS.gridColor },
                    ticks: { callback: v => v + '%' },
                },
                y: { grid: { display: false } },
            },
        }
    });
}

// ── Income Percentiles (Area Chart) ──
function renderPercentiles() {
    const d = chartData.income_percentiles;
    if (!d) return;

    new Chart(document.getElementById('chartPercentiles'), {
        type: 'line',
        data: {
            labels: d.labels,
            datasets: [{
                label: 'Thu nhập (USD)',
                data: d.values,
                borderColor: CHART_COLORS.primary,
                backgroundColor: hexToRgba(CHART_COLORS.primary, 0.15),
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointHoverRadius: 7,
                pointBackgroundColor: CHART_COLORS.primary,
                pointBorderColor: '#0a0e1a',
                pointBorderWidth: 2,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: { label: ctx => `$${ctx.raw.toLocaleString()}` },
                },
            },
            scales: {
                x: { grid: { color: CHART_COLORS.gridColor } },
                y: {
                    grid: { color: CHART_COLORS.gridColor },
                    ticks: { callback: v => '$' + fmtNum(v) },
                },
            },
        }
    });
}

// ── Doughnut Chart ──
function renderDoughnut(canvasId, data) {
    if (!data) return;
    new Chart(document.getElementById(canvasId), {
        type: 'doughnut',
        data: {
            labels: data.labels,
            datasets: [{
                data: data.values,
                backgroundColor: CHART_COLORS.palette.slice(0, data.labels.length),
                borderColor: '#0a0e1a',
                borderWidth: 3,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            cutout: '55%',
            plugins: {
                legend: { position: 'bottom', labels: { padding: 14, font: { size: 11 } } },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const total = ctx.dataset.data.reduce((a,b) => a+b, 0);
                            const pct = ((ctx.raw / total) * 100).toFixed(1);
                            return `${ctx.label}: ${ctx.raw.toLocaleString()} (${pct}%)`;
                        }
                    }
                },
            },
        }
    });
}

// ============================================================
// Prediction Engine (unchanged logic)
// ============================================================
function predict(formData) {
    if (!modelParams) return null;
    const p = modelParams;
    const features = [];

    for (const col of p.feature_columns) {
        if (col.endsWith('_encoded')) {
            const orig = col.replace('_encoded', '');
            features.push(p.encoding_maps[orig]?.[formData[orig]] ?? 0);
        } else if (col === 'Age_WorkExp_Ratio') {
            features.push(formData.Age / (formData.Work_Experience + 1));
        } else if (col === 'Dependents_per_HHSize') {
            features.push(formData.Number_of_Dependents / formData.Household_Size);
        } else {
            features.push(formData[col] || 0);
        }
    }

    const scaled = features.map((v, i) => (v - p.scaler_mean[i]) / p.scaler_scale[i]);
    let pred = p.intercept;
    scaled.forEach((v, i) => pred += v * p.coefficients[i]);
    if (p.use_log_transform) pred = Math.expm1(pred);
    return { income: Math.max(pred, 0), features, scaledFeatures: scaled };
}

// ============================================================
// Form Handling
// ============================================================
function initFormControls() {
    document.querySelectorAll('.range-input').forEach(input => {
        const valEl = document.getElementById(input.id + 'Value');
        if (valEl) {
            input.addEventListener('input', () => { valEl.textContent = input.value; updateRangeTrack(input); });
            updateRangeTrack(input);
        }
    });
    document.getElementById('predictionForm')?.addEventListener('submit', handlePrediction);
}

function updateRangeTrack(input) {
    const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.background = `linear-gradient(to right, #6366f1 0%, #8b5cf6 ${pct}%, rgba(99,102,241,0.2) ${pct}%)`;
}

function handlePrediction(e) {
    e.preventDefault();
    if (!isModelLoaded) { alert('Mô hình đang tải…'); return; }

    const fd = {
        Age: +document.getElementById('age').value,
        Number_of_Dependents: +document.getElementById('dependents').value,
        Work_Experience: +document.getElementById('workExperience').value,
        Household_Size: +document.getElementById('householdSize').value,
        Education_Level: document.getElementById('education').value,
        Occupation: document.getElementById('occupation').value,
        Location: document.getElementById('location').value,
        Marital_Status: document.getElementById('maritalStatus').value,
        Employment_Status: document.getElementById('employmentStatus').value,
        Homeownership_Status: document.getElementById('homeownership').value,
        Type_of_Housing: document.getElementById('housingType').value,
        Gender: document.getElementById('gender').value,
        Primary_Mode_of_Transportation: document.getElementById('transportation').value,
    };

    const res = predict(fd);
    if (res) displayResult(res, fd);
}

// ============================================================
// Display Result
// ============================================================
function displayResult(result, formData) {
    document.getElementById('resultPlaceholder').style.display = 'none';
    const content = document.getElementById('resultContent');
    content.style.display = 'block';
    content.style.animation = 'none';
    content.offsetHeight;
    content.style.animation = 'fadeInUp 0.5s ease forwards';

    setText('modelUsed', 'Random Forest (Ridge web)');

    const income = Math.round(result.income);
    animateIncome(document.getElementById('predictedIncome'), income);
    setText('predictedMonthly', `~$${fmtNum(Math.round(income / 12))}/tháng`);

    const median = modelMetadata?.income_stats?.median || 72943;
    const pctile = estimatePercentile(income, modelMetadata?.income_stats);

    document.getElementById('resultBreakdown').innerHTML = `
        <div class="breakdown-item"><div class="breakdown-label">So với trung vị</div>
            <div class="breakdown-value ${income > median ? 'positive' : 'neutral'}">${income > median ? '+' : ''}${((income/median-1)*100).toFixed(1)}%</div></div>
        <div class="breakdown-item"><div class="breakdown-label">Phân vị</div>
            <div class="breakdown-value neutral">Top ${pctile}%</div></div>
        <div class="breakdown-item"><div class="breakdown-label">Hàng ngày</div>
            <div class="breakdown-value neutral">~$${fmtNum(Math.round(income/365))}</div></div>
        <div class="breakdown-item"><div class="breakdown-label">Hàng tuần</div>
            <div class="breakdown-value neutral">~$${fmtNum(Math.round(income/52))}</div></div>`;

    if (modelParams) {
        const contribs = modelParams.feature_columns.map((c, i) => ({
            name: c, value: Math.abs(result.scaledFeatures[i] * modelParams.coefficients[i]),
            raw: result.scaledFeatures[i] * modelParams.coefficients[i],
        })).sort((a,b) => b.value - a.value);
        const mx = contribs[0].value;
        const names = { Age:'Tuổi', Work_Experience:'Kinh nghiệm', Education_Level_encoded:'Học vấn',
            Occupation_encoded:'Ngành nghề', Location_encoded:'Khu vực', Household_Size:'Quy mô HGĐ',
            Employment_Status_encoded:'Việc làm', Homeownership_Status_encoded:'Sở hữu nhà',
            Gender_encoded:'Giới tính', Age_WorkExp_Ratio:'Tuổi/KN', Dependents_per_HHSize:'PT/QM',
            Number_of_Dependents:'Phụ thuộc', Type_of_Housing_encoded:'Loại nhà',
            Marital_Status_encoded:'Hôn nhân', Primary_Mode_of_Transportation_encoded:'Phương tiện' };

        document.getElementById('resultFactors').innerHTML = `<h4>Yếu tố ảnh hưởng chính</h4>` +
            contribs.slice(0,5).map(f => `<div class="factor-item">
                <span class="factor-name">${names[f.name]||f.name}</span>
                <div class="factor-bar-bg"><div class="factor-bar-fill" style="width:${(f.value/mx*100).toFixed(0)}%;background:${f.raw>0?'#10b981':'#ef4444'}"></div></div>
                <span class="factor-value">${f.raw>0?'+':'-'}</span></div>`).join('');
    }

    if (window.innerWidth <= 1024) document.getElementById('resultPanel').scrollIntoView({ behavior:'smooth', block:'start' });
}

// ============================================================
// Navigation
// ============================================================
function initNavigation() {
    const navbar = document.getElementById('navbar');
    const toggle = document.getElementById('mobileToggle');
    const links = document.querySelector('.nav-links');

    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 50);
        updateActiveNav();
    });
    toggle?.addEventListener('click', () => links.classList.toggle('open'));

    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', e => {
            const id = link.getAttribute('href');
            if (id.startsWith('#')) { e.preventDefault(); document.querySelector(id)?.scrollIntoView({behavior:'smooth'}); links.classList.remove('open'); }
        });
    });
}

function updateActiveNav() {
    let cur = 'hero';
    ['hero','predictor','insights','model'].forEach(id => {
        const el = document.getElementById(id);
        if (el && el.getBoundingClientRect().top <= 150) cur = id;
    });
    document.querySelectorAll('.nav-link').forEach(l => l.classList.toggle('active', l.dataset.section === cur));
}

function initScrollAnimations() {
    const obs = new IntersectionObserver(entries => {
        entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
    document.querySelectorAll('.animate-in, .chart-card, .method-step, .stat-card').forEach(el => obs.observe(el));
}

// ============================================================
// Utilities
// ============================================================
function fmtNum(n) {
    if (n == null) return '0';
    if (n >= 1e6) return (n/1e6).toFixed(1)+'M';
    if (n >= 1e3) return (n/1e3).toFixed(n>=1e4?0:1)+'K';
    return Math.round(n).toLocaleString();
}

function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }

function animateIncome(el, target) {
    const dur = 1200, start = performance.now();
    (function tick(now) {
        const p = Math.min((now-start)/dur, 1);
        el.textContent = '$' + Math.round(target * (1 - Math.pow(1-p, 4))).toLocaleString();
        if (p < 1) requestAnimationFrame(tick);
    })(start);
}

function estimatePercentile(inc, s) {
    if (!s) return 50;
    if (inc<=s.q25) return Math.round(75+(1-inc/s.q25)*25);
    if (inc<=s.median) return Math.round(50+(1-(inc-s.q25)/(s.median-s.q25))*25);
    if (inc<=s.q75) return Math.round(25+(1-(inc-s.median)/(s.q75-s.median))*25);
    return Math.max(1, Math.round(25*(1-(inc-s.q75)/(s.max-s.q75))));
}

function hexToRgba(hex, a) {
    const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    return `rgba(${r},${g},${b},${a})`;
}

function createGradientBar(canvasId, c1, c2) {
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return c1;
    const g = ctx.createLinearGradient(0, 0, 0, 400);
    g.addColorStop(0, c1); g.addColorStop(1, c2);
    return g;
}

// ── Fallback ──
function useFallbackData() {
    modelParams = {
        type:'ridge_log', use_log_transform:true,
        coefficients:[.01,.02,-.01,.01,.03,.05,-.02,.01,.02,.04,-.01,.005,-.01,.015,.01],
        intercept:11.2,
        scaler_mean:[44,2.5,25,4,1.5,2,1,.8,.6,.4,.8,.5,1.5,2.5,.5],
        scaler_scale:[15,1.7,14.5,2,1.2,1.5,.9,.8,.8,.5,.8,.5,1.1,3,.3],
        feature_columns:['Age','Number_of_Dependents','Work_Experience','Household_Size',
            'Education_Level_encoded','Occupation_encoded','Location_encoded',
            'Marital_Status_encoded','Employment_Status_encoded',
            'Homeownership_Status_encoded','Type_of_Housing_encoded',
            'Gender_encoded','Primary_Mode_of_Transportation_encoded',
            'Age_WorkExp_Ratio','Dependents_per_HHSize'],
        encoding_maps:{
            Education_Level:{"Bachelor's":0,Doctorate:1,"High School":2,"Master's":3},
            Occupation:{Education:0,Finance:1,Healthcare:2,Others:3,Technology:4},
            Location:{Rural:0,Suburban:1,Urban:2},
            Marital_Status:{Divorced:0,Married:1,Single:2},
            Employment_Status:{"Full-time":0,"Part-time":1,"Self-employed":2},
            Homeownership_Status:{Own:0,Rent:1},
            Type_of_Housing:{Apartment:0,"Single-family home":1,Townhouse:2},
            Gender:{Female:0,Male:1},
            Primary_Mode_of_Transportation:{Biking:0,Car:1,"Public transit":2,Walking:3},
        },
    };
    modelMetadata = {
        best_model_name:'Random Forest (Log)', web_model_name:'Random Forest (Ridge web)',
        metrics:{R2:0.15,R2_log:0.1472,MAE:763300,RMSE:1846157,MAPE:140,CV_R2_Mean:0.1527},
        income_stats:{mean:816838,median:72943,min:31044,max:9992571,std:1821089,q25:68446,q75:350668},
        dataset_info:{total_rows:10000,features:15,train_size:8000,test_size:2000},
        feature_importance:{Age_WorkExp_Ratio:.1708,Work_Experience:.134,Age:.1226,Dependents_per_HHSize:.0888,
            Occupation_encoded:.0781,Household_Size:.0558,Employment_Status_encoded:.0539,
            Education_Level_encoded:.0487,Number_of_Dependents:.0476,Type_of_Housing_encoded:.045},
    };
    isModelLoaded = true;
    populateHeroStats();
}
