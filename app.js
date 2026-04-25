/**
 * HealthScan ML — app.js
 * Frontend logic: API calls, result rendering, UX helpers
 */

const API = "http://localhost:5000/api";

// ─────────────────────────────────────────────
// TAB NAVIGATION
// ─────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById(`tab-${name}`).classList.add('active');
  document.querySelectorAll('.nav-tab').forEach(t => {
    if (t.getAttribute('onclick').includes(name)) t.classList.add('active');
  });
}

// ─────────────────────────────────────────────
// LOADING OVERLAY
// ─────────────────────────────────────────────
function showLoading() { document.getElementById('loading-overlay').classList.remove('hidden'); }
function hideLoading() { document.getElementById('loading-overlay').classList.add('hidden'); }

// ─────────────────────────────────────────────
// SERVER STATUS CHECK
// ─────────────────────────────────────────────
async function checkServer() {
  try {
    const resp = await fetch(`${API}/health`);
    const data = await resp.json();
    document.getElementById('accuracy-badge').textContent = `Accuracy: ${data.model_accuracy}`;
  } catch {
    document.getElementById('accuracy-badge').textContent = 'Server Offline';
    document.getElementById('accuracy-badge').style.color = '#ef4444';
  }
}

// ─────────────────────────────────────────────
// BMI HINT
// ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkServer();
  const bmiInput = document.getElementById('bmi');
  if (bmiInput) {
    bmiInput.addEventListener('input', () => {
      const bmi = parseFloat(bmiInput.value);
      const hint = document.getElementById('bmi-hint');
      if (!bmi) { hint.textContent = ''; return; }
      if (bmi < 18.5) { hint.textContent = 'Underweight'; hint.style.color = '#60a5fa'; }
      else if (bmi < 25) { hint.textContent = 'Normal weight ✓'; hint.style.color = '#10b981'; }
      else if (bmi < 30) { hint.textContent = 'Overweight'; hint.style.color = '#f59e0b'; }
      else { hint.textContent = 'Obese'; hint.style.color = '#ef4444'; }
    });
  }
});

// ─────────────────────────────────────────────
// DEMO DATA PROFILES
// ─────────────────────────────────────────────
const demos = {
  healthy: {
    age: 32, bmi: 22.5, glucose: 88, systolic_bp: 115, diastolic_bp: 75,
    cholesterol: 170, hdl: 65, ldl: 90, hemoglobin: 14.5, heart_rate: 68,
    smoker: false, diabetic: false, exercise_days: 5
  },
  risk: {
    age: 58, bmi: 33.4, glucose: 148, systolic_bp: 155, diastolic_bp: 98,
    cholesterol: 265, hdl: 32, ldl: 185, hemoglobin: 10.2, heart_rate: 95,
    smoker: true, diabetic: true, exercise_days: 1
  }
};

function loadDemo(type) {
  const d = demos[type];
  document.getElementById('age').value           = d.age;
  document.getElementById('bmi').value           = d.bmi;
  document.getElementById('glucose').value       = d.glucose;
  document.getElementById('systolic_bp').value   = d.systolic_bp;
  document.getElementById('diastolic_bp').value  = d.diastolic_bp;
  document.getElementById('cholesterol').value   = d.cholesterol;
  document.getElementById('hdl').value           = d.hdl;
  document.getElementById('ldl').value           = d.ldl;
  document.getElementById('hemoglobin').value    = d.hemoglobin;
  document.getElementById('heart_rate').value    = d.heart_rate;
  document.getElementById('smoker').checked      = d.smoker;
  document.getElementById('diabetic').checked    = d.diabetic;
  document.getElementById('exercise_days').value = d.exercise_days;
  // trigger bmi hint
  document.getElementById('bmi').dispatchEvent(new Event('input'));
}

// ─────────────────────────────────────────────
// VITALS PREDICTION
// ─────────────────────────────────────────────
async function predictRisk() {
  const fields = ['age','bmi','glucose','systolic_bp','diastolic_bp',
                  'cholesterol','hdl','ldl','hemoglobin','heart_rate','exercise_days'];

  const payload = {};
  for (const f of fields) {
    const val = parseFloat(document.getElementById(f).value);
    if (isNaN(val)) { showError('vitals-result', `Please fill in "${f.replace(/_/g,' ')}"`); return; }
    payload[f] = val;
  }
  payload.smoker   = document.getElementById('smoker').checked   ? 1 : 0;
  payload.diabetic = document.getElementById('diabetic').checked ? 1 : 0;

  showLoading();
  try {
    const resp = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    hideLoading();
    if (data.error) { showError('vitals-result', data.error); return; }
    renderVitalsResult(data);
  } catch (err) {
    hideLoading();
    showError('vitals-result', 'Cannot connect to server. Make sure the Python backend is running on port 5000.');
  }
}

function renderVitalsResult(data) {
  const panel = document.getElementById('vitals-result');
  panel.classList.remove('hidden');

  const riskClass = data.risk_level.toLowerCase();
  const c = data.confidence;

  const diseasesHtml = Object.entries(data.disease_risks).map(([name, pct]) => {
    const cls = pct >= 60 ? 'pct-high' : pct >= 30 ? 'pct-medium' : 'pct-low';
    const barCls = pct >= 60 ? '#ef4444' : pct >= 30 ? '#f59e0b' : '#10b981';
    return `
      <div class="disease-card">
        <div class="disease-name">${name}</div>
        <div class="disease-pct ${cls}">${pct}%</div>
        <div class="disease-bar-bg">
          <div class="disease-bar-fill" style="width:${pct}%;background:${barCls}"></div>
        </div>
      </div>`;
  }).join('');

  const rec = data.recommendations;

  const recSections = [
    { key: 'urgent',     title: '⚠️ Urgent Actions',       cls: 'urgent' },
    { key: 'lifestyle',  title: '🌿 Lifestyle',             cls: '' },
    { key: 'diet',       title: '🥗 Diet',                  cls: '' },
    { key: 'exercise',   title: '🏃 Exercise',              cls: '' },
    { key: 'monitoring', title: '📊 Health Monitoring',     cls: '' },
  ].filter(s => rec[s.key] && rec[s.key].length > 0)
   .map(s => `
    <div class="rec-card ${s.cls}">
      <div class="rec-title">${s.title}</div>
      ${rec[s.key].map(r => `<div class="rec-item">${r}</div>`).join('')}
    </div>`).join('');

  panel.innerHTML = `
    <div class="risk-header">
      <div>
        <div style="font-size:13px;color:var(--muted);margin-bottom:6px">Overall Risk Level</div>
        <div class="risk-badge risk-${riskClass}">${data.risk_level} Risk</div>
        <div style="font-size:12px;color:var(--muted);margin-top:6px">BMI: ${document.getElementById('bmi').value} — ${data.bmi_category}</div>
      </div>
      <div class="confidence-bars">
        <div style="font-size:12px;color:var(--muted);margin-bottom:8px">Model Confidence (${data.model_accuracy})</div>
        ${confBar('Low', c.low, 'conf-low')}
        ${confBar('Medium', c.medium, 'conf-medium')}
        ${confBar('High', c.high, 'conf-high')}
      </div>
    </div>

    <div class="label">Disease Risk Assessment</div>
    <div class="disease-grid">${diseasesHtml}</div>

    <div class="label">Personalised Recommendations</div>
    <div class="rec-sections">${recSections}</div>
  `;
}

function confBar(label, pct, cls) {
  return `
    <div class="conf-row">
      <span class="conf-label">${label}</span>
      <div class="conf-bar-bg"><div class="conf-bar-fill ${cls}" style="width:${pct}%"></div></div>
      <span class="conf-pct">${pct}%</span>
    </div>`;
}

// ─────────────────────────────────────────────
// FOOD ANALYSIS
// ─────────────────────────────────────────────
function quickFood(name) {
  document.getElementById('food-input').value = name;
  analyzeFood();
}

async function analyzeFood() {
  const name = document.getElementById('food-input').value.trim();
  if (!name) return;

  showLoading();
  try {
    const resp = await fetch(`${API}/food?name=${encodeURIComponent(name)}`);
    const data = await resp.json();
    hideLoading();
    if (data.error) {
      showError('food-result', data.error + (data.available_foods ? `<br><small>Available: ${data.available_foods.join(', ')}</small>` : ''));
      return;
    }
    renderFoodResult(data);
  } catch {
    hideLoading();
    showError('food-result', 'Cannot connect to server. Please start the Python backend.');
  }
}

function renderFoodResult(d) {
  const panel = document.getElementById('food-result');
  panel.classList.remove('hidden');

  const scoreClass = d.health_score >= 8 ? 'score-high' : d.health_score >= 5 ? 'score-medium' : 'score-low';
  const scoreLabel = d.health_score >= 8 ? 'Healthy' : d.health_score >= 5 ? 'Moderate' : 'Avoid';
  const giColor    = d.glycemic_index >= 70 ? '#ef4444' : d.glycemic_index >= 56 ? '#f59e0b' : '#10b981';

  const benefitsHtml = (d.benefits||[]).map(b => `<span class="tag tag-green">${b}</span>`).join('');
  const risksHtml    = (d.risks||[]).map(r => `<span class="tag tag-red">${r}</span>`).join('');
  const recoHtml     = (d.recommended_for||[]).map(r => `<span class="tag tag-blue">${r}</span>`).join('');
  const avoidHtml    = (d.avoid_if||[]).map(a => `<span class="tag tag-red">${a}</span>`).join('');

  panel.innerHTML = `
    <div class="food-card-result">
      <div class="food-score-circle ${scoreClass}">
        <span>${d.health_score}</span>
        <span style="font-size:10px;font-weight:400">${scoreLabel}</span>
      </div>
      <div>
        <h3 style="font-family:var(--font-head);font-size:22px;margin-bottom:4px">${capitalize(d.food)}</h3>
        <div style="color:var(--muted);font-size:13px">${d.category} · Per ${d.portion}</div>
      </div>
    </div>

    <div class="nutrition-grid" style="margin-top:20px">
      ${nutriCell(d.calories, 'Calories', 'kcal')}
      ${nutriCell(d.protein,  'Protein',  'g')}
      ${nutriCell(d.carbs,    'Carbs',    'g')}
      ${nutriCell(d.fat,      'Fat',      'g')}
    </div>

    <div style="background:var(--surface2);border-radius:10px;padding:12px;margin-bottom:16px">
      <span style="font-size:13px;color:var(--muted)">Glycemic Index: </span>
      <strong style="color:${giColor}">${d.glycemic_index === 0 ? 'N/A (no carbs)' : d.glycemic_index + ' — ' + giLabel(d.glycemic_index)}</strong>
    </div>

    <div class="label">Benefits</div>
    <div class="tag-list">${benefitsHtml}</div>

    <div class="label">Potential Risks</div>
    <div class="tag-list">${risksHtml}</div>

    <div class="label">Recommended For</div>
    <div class="tag-list">${recoHtml}</div>

    <div class="label">Avoid If</div>
    <div class="tag-list">${avoidHtml}</div>
  `;
}

function nutriCell(val, label, unit) {
  return `<div class="nutr-cell"><div class="nutr-val">${val}</div><div class="nutr-label">${label} (${unit})</div></div>`;
}

function giLabel(gi) {
  if (gi <= 55) return 'Low GI ✓';
  if (gi <= 69) return 'Medium GI';
  return 'High GI ⚠️';
}

// ─────────────────────────────────────────────
// MEDICINE ANALYSIS
// ─────────────────────────────────────────────
function quickMed(name) {
  document.getElementById('medicine-input').value = name;
  analyzeMedicine();
}

async function analyzeMedicine() {
  const name = document.getElementById('medicine-input').value.trim();
  if (!name) return;

  showLoading();
  try {
    const resp = await fetch(`${API}/medicine?name=${encodeURIComponent(name)}`);
    const data = await resp.json();
    hideLoading();
    if (data.error) {
      showError('medicine-result', data.error + (data.available_medicines ? `<br><small>Available: ${data.available_medicines.join(', ')}</small>` : ''));
      return;
    }
    renderMedicineResult(data);
  } catch {
    hideLoading();
    showError('medicine-result', 'Cannot connect to server. Please start the Python backend.');
  }
}

function renderMedicineResult(d) {
  const panel = document.getElementById('medicine-result');
  panel.classList.remove('hidden');

  const riskColor = d.risk_level === 'Low' ? 'var(--success)' :
                    d.risk_level === 'Medium' ? 'var(--warning)' : 'var(--danger)';

  panel.innerHTML = `
    <h3 style="font-family:var(--font-head);font-size:24px;margin-bottom:8px">${capitalize(d.medicine)}</h3>
    <div class="med-meta">
      <span class="med-badge">${d.class}</span>
      <span class="med-badge">${d.type}</span>
      <span class="med-badge" style="color:${riskColor};border-color:${riskColor}">Risk: ${d.risk_level}</span>
    </div>

    <div style="background:var(--surface2);border-radius:10px;padding:14px;margin-bottom:16px;font-size:13px;color:var(--muted)">
      <strong style="color:var(--text)">Mechanism:</strong> ${d.mechanism}
    </div>

    <div class="info-grid">
      <div class="info-block">
        <div class="info-block-title">✅ Used For</div>
        <ul>${(d.used_for||[]).map(i=>`<li>${i}</li>`).join('')}</ul>
      </div>
      <div class="info-block">
        <div class="info-block-title">⚠️ Common Side Effects</div>
        <ul>${(d.common_side_effects||[]).map(i=>`<li>${i}</li>`).join('')}</ul>
      </div>
      <div class="info-block">
        <div class="info-block-title">🚫 Drug Interactions</div>
        <ul>${(d.serious_interactions||[]).map(i=>`<li>${i}</li>`).join('')}</ul>
      </div>
      <div class="info-block">
        <div class="info-block-title">❌ Avoid If</div>
        <ul>${(d.avoid_if||[]).map(i=>`<li>${i}</li>`).join('')}</ul>
      </div>
      <div class="info-block">
        <div class="info-block-title">🔬 Monitor</div>
        <ul>${(d.monitor||[]).map(i=>`<li>${i}</li>`).join('')}</ul>
      </div>
      <div class="info-block">
        <div class="info-block-title">🍽️ Take With</div>
        <ul><li>${d.take_with}</li></ul>
      </div>
    </div>
  `;
}

// ─────────────────────────────────────────────
// FULL SCAN — Tag inputs
// ─────────────────────────────────────────────
let foodTags = [];
let medTags  = [];

function addFoodTag() {
  const val = document.getElementById('food-tag-input').value.trim();
  if (val && !foodTags.includes(val)) {
    foodTags.push(val);
    renderTags('food-tags', foodTags, removeFoodTag);
    document.getElementById('food-tag-input').value = '';
  }
}

function addFoodTagVal(val) {
  if (!foodTags.includes(val)) {
    foodTags.push(val);
    renderTags('food-tags', foodTags, removeFoodTag);
  }
}

function removeFoodTag(val) {
  foodTags = foodTags.filter(t => t !== val);
  renderTags('food-tags', foodTags, removeFoodTag);
}

function addMedTag() {
  const val = document.getElementById('med-tag-input').value.trim();
  if (val && !medTags.includes(val)) {
    medTags.push(val);
    renderTags('med-tags', medTags, removeMedTag);
    document.getElementById('med-tag-input').value = '';
  }
}

function addMedTagVal(val) {
  if (!medTags.includes(val)) {
    medTags.push(val);
    renderTags('med-tags', medTags, removeMedTag);
  }
}

function removeMedTag(val) {
  medTags = medTags.filter(t => t !== val);
  renderTags('med-tags', medTags, removeMedTag);
}

function renderTags(containerId, tags, removeFn) {
  const container = document.getElementById(containerId);
  container.innerHTML = tags.map(t => `
    <span class="tag-item">${t}
      <button onclick="${removeFn.name}('${t}')">×</button>
    </span>`).join('');
}

// ─────────────────────────────────────────────
// FULL SCAN
// ─────────────────────────────────────────────
async function runFullScan() {
  const vitals = {
    age:           parseFloat(document.getElementById('f-age').value)     || 45,
    bmi:           parseFloat(document.getElementById('f-bmi').value)     || 27,
    glucose:       parseFloat(document.getElementById('f-glucose').value) || 105,
    systolic_bp:   parseFloat(document.getElementById('f-sbp').value)     || 135,
    diastolic_bp:  parseFloat(document.getElementById('f-dbp').value)     || 85,
    cholesterol:   parseFloat(document.getElementById('f-chol').value)    || 210,
    hdl:           parseFloat(document.getElementById('f-hdl').value)     || 45,
    ldl:           parseFloat(document.getElementById('f-ldl').value)     || 140,
    hemoglobin:    parseFloat(document.getElementById('f-hb').value)      || 13.5,
    heart_rate:    parseFloat(document.getElementById('f-hr').value)      || 75,
    smoker:        document.getElementById('f-smoker').checked   ? 1 : 0,
    diabetic:      document.getElementById('f-diabetic').checked ? 1 : 0,
    exercise_days: parseFloat(document.getElementById('f-ex').value)      || 2
  };

  showLoading();
  try {
    const resp = await fetch(`${API}/full_diagnosis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vitals, foods: foodTags, medicines: medTags })
    });
    const data = await resp.json();
    hideLoading();
    if (data.error) { showError('full-result', data.error); return; }
    renderFullResult(data);
  } catch {
    hideLoading();
    showError('full-result', 'Cannot connect to server. Please start the Python backend.');
  }
}

function renderFullResult(data) {
  const panel = document.getElementById('full-result');
  panel.classList.remove('hidden');

  const hp   = data.health_prediction;
  const riskClass = hp.risk_level.toLowerCase();
  const c    = hp.confidence;

  // Disease risks mini
  const diseasesHtml = Object.entries(hp.disease_risks).map(([name, pct]) => {
    const barCls = pct >= 60 ? '#ef4444' : pct >= 30 ? '#f59e0b' : '#10b981';
    return `<div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px">
        <span>${name}</span><span style="color:${barCls}">${pct}%</span>
      </div>
      <div class="disease-bar-bg"><div class="disease-bar-fill" style="width:${pct}%;background:${barCls}"></div></div>
    </div>`;
  }).join('');

  // Food summary
  const foodsHtml = Object.entries(data.food_analysis || {}).length === 0
    ? '<p style="color:var(--muted);font-size:13px">No foods added.</p>'
    : Object.entries(data.food_analysis).map(([name, f]) => {
        const sc = f.health_score >= 8 ? '#10b981' : f.health_score >= 5 ? '#f59e0b' : '#ef4444';
        return `<div style="background:var(--surface2);border-radius:10px;padding:12px;margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <strong>${capitalize(name)}</strong>
            <span style="color:${sc};font-weight:700">Score: ${f.health_score}/10</span>
          </div>
          <div style="font-size:12px;color:var(--muted)">${f.calories} kcal · GI: ${f.glycemic_index||'N/A'}</div>
          <div style="font-size:12px;margin-top:4px">${(f.benefits||[]).slice(0,2).join(' · ')}</div>
        </div>`;
      }).join('');

  // Medicine summary
  const medsHtml = Object.entries(data.medicine_analysis || {}).length === 0
    ? '<p style="color:var(--muted);font-size:13px">No medicines added.</p>'
    : Object.entries(data.medicine_analysis).map(([name, m]) => {
        const rc = m.risk_level === 'Low' ? '#10b981' : m.risk_level === 'Medium' ? '#f59e0b' : '#ef4444';
        return `<div style="background:var(--surface2);border-radius:10px;padding:12px;margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <strong>${capitalize(name)}</strong>
            <span style="color:${rc};font-weight:700">${m.risk_level} Risk</span>
          </div>
          <div style="font-size:12px;color:var(--muted)">${m.class} · ${m.type}</div>
          <div style="font-size:12px;margin-top:4px;color:var(--muted)">Side effects: ${(m.common_side_effects||[]).slice(0,2).join(', ')}</div>
        </div>`;
      }).join('');

  // Recommendations
  const rec = hp.recommendations;
  const urgentHtml = (rec.urgent||[]).length > 0
    ? `<div class="rec-card urgent"><div class="rec-title">⚠️ Urgent</div>${rec.urgent.map(r=>`<div class="rec-item">${r}</div>`).join('')}</div>`
    : '';

  panel.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px">
      <!-- Health Risk Summary -->
      <div>
        <div class="label">Overall Risk</div>
        <div class="risk-badge risk-${riskClass}" style="margin-bottom:16px">${hp.risk_level} Risk</div>
        <div style="margin-bottom:16px">
          ${confBar('Low', c.low, 'conf-low')}
          ${confBar('Medium', c.medium, 'conf-medium')}
          ${confBar('High', c.high, 'conf-high')}
        </div>
        <div class="label">Disease Risks</div>
        ${diseasesHtml}
      </div>

      <!-- Food & Medicine -->
      <div>
        <div class="label">Food Analysis</div>
        ${foodsHtml}
        <div class="label" style="margin-top:16px">Medicine Review</div>
        ${medsHtml}
      </div>
    </div>

    <div class="label" style="margin-top:20px">Key Recommendations</div>
    <div class="rec-sections">
      ${urgentHtml}
      ${buildRecCard('🌿 Lifestyle', rec.lifestyle||[], '')}
      ${buildRecCard('🥗 Diet', rec.diet||[], '')}
      ${buildRecCard('🏃 Exercise', rec.exercise||[], '')}
    </div>
  `;
}

function buildRecCard(title, items, cls) {
  if (!items.length) return '';
  return `<div class="rec-card ${cls}">
    <div class="rec-title">${title}</div>
    ${items.map(r=>`<div class="rec-item">${r}</div>`).join('')}
  </div>`;
}

// ─────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────
function showError(panelId, message) {
  const panel = document.getElementById(panelId);
  panel.classList.remove('hidden');
  panel.innerHTML = `
    <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:10px;padding:16px;color:#ef4444">
      ⚠️ ${message}
    </div>`;
}

function capitalize(str) {
  return str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// Enter key on inputs
document.addEventListener('DOMContentLoaded', () => {
  const foodInput = document.getElementById('food-input');
  const medInput  = document.getElementById('medicine-input');
  if (foodInput) foodInput.addEventListener('keydown', e => { if(e.key==='Enter') analyzeFood(); });
  if (medInput)  medInput.addEventListener('keydown',  e => { if(e.key==='Enter') analyzeMedicine(); });
  const foodTag = document.getElementById('food-tag-input');
  const medTag  = document.getElementById('med-tag-input');
  if (foodTag) foodTag.addEventListener('keydown', e => { if(e.key==='Enter') addFoodTag(); });
  if (medTag)  medTag.addEventListener('keydown',  e => { if(e.key==='Enter') addMedTag(); });
});