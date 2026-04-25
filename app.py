"""
HealthScan ML Backend - Flask API
Real-time health risk prediction with food & medicine diagnosis
"""

from flask import Flask, request, jsonify, Response
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─────────────────────────────────────────────
# CORS helper (no flask-cors needed)
# ─────────────────────────────────────────────
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.after_request
def after_request(response):
    return add_cors(response)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        resp = Response()
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return resp


# ─────────────────────────────────────────────
# SYNTHETIC TRAINING DATA GENERATION
# ─────────────────────────────────────────────
def generate_training_data(n=2000):
    np.random.seed(42)

    age       = np.random.randint(18, 85, n)
    bmi       = np.round(np.random.uniform(15, 45, n), 1)
    glucose   = np.random.randint(60, 300, n)
    bp_sys    = np.random.randint(80, 200, n)
    bp_dia    = np.random.randint(50, 130, n)
    chol      = np.random.randint(120, 320, n)
    hdl       = np.random.randint(20, 100, n)
    ldl       = np.random.randint(50, 220, n)
    hb        = np.round(np.random.uniform(7, 18, n), 1)
    heart_rate= np.random.randint(45, 140, n)
    smoker    = np.random.randint(0, 2, n)
    diabetic  = np.random.randint(0, 2, n)
    exercise  = np.random.randint(0, 8, n)  # days/week

    # Risk score (0=Low, 1=Medium, 2=High)
    risk = np.zeros(n, dtype=int)
    for i in range(n):
        score = 0
        if age[i] > 55:           score += 2
        elif age[i] > 40:         score += 1
        if bmi[i] > 30:           score += 2
        elif bmi[i] > 25:         score += 1
        if glucose[i] > 140:      score += 3
        elif glucose[i] > 100:    score += 1
        if bp_sys[i] > 140:       score += 2
        elif bp_sys[i] > 120:     score += 1
        if chol[i] > 240:         score += 2
        elif chol[i] > 200:       score += 1
        if ldl[i] > 160:          score += 2
        elif ldl[i] > 130:        score += 1
        if hdl[i] < 40:           score += 2
        if hb[i] < 10:            score += 2
        elif hb[i] < 12:          score += 1
        if smoker[i]:             score += 2
        if diabetic[i]:           score += 2
        if exercise[i] < 2:       score += 1
        if score <= 3:            risk[i] = 0
        elif score <= 7:          risk[i] = 1
        else:                     risk[i] = 2

    df = pd.DataFrame({
        'age': age, 'bmi': bmi, 'glucose': glucose,
        'systolic_bp': bp_sys, 'diastolic_bp': bp_dia,
        'cholesterol': chol, 'hdl': hdl, 'ldl': ldl,
        'hemoglobin': hb, 'heart_rate': heart_rate,
        'smoker': smoker, 'diabetic': diabetic,
        'exercise_days': exercise, 'risk': risk
    })
    return df


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
df = generate_training_data(2000)
FEATURES = ['age','bmi','glucose','systolic_bp','diastolic_bp',
            'cholesterol','hdl','ldl','hemoglobin','heart_rate',
            'smoker','diabetic','exercise_days']

X = df[FEATURES]
y = df['risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                   max_depth=4, random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained — accuracy: {accuracy:.2%}")


# ─────────────────────────────────────────────
# FOOD DATABASE
# ─────────────────────────────────────────────
FOOD_DB = {
    "rice": {
        "category": "Carbohydrate", "calories": 206, "protein": 4.3,
        "fat": 0.4, "carbs": 45, "glycemic_index": 73,
        "health_score": 6,
        "benefits": ["Energy source", "Low fat", "Gluten-free"],
        "risks": ["High glycemic index", "Spikes blood sugar if eaten in excess"],
        "recommended_for": ["Athletes", "Active individuals"],
        "avoid_if": ["Diabetes", "Obesity", "Insulin resistance"],
        "portion": "1/2 cup cooked"
    },
    "apple": {
        "category": "Fruit", "calories": 95, "protein": 0.5,
        "fat": 0.3, "carbs": 25, "glycemic_index": 36,
        "health_score": 9,
        "benefits": ["Rich in fiber", "Antioxidants", "Heart healthy", "Low glycemic"],
        "risks": ["High sugar for diabetics in large quantities"],
        "recommended_for": ["Everyone", "Diabetics (moderate)", "Heart disease"],
        "avoid_if": ["Apple allergy"],
        "portion": "1 medium apple"
    },
    "salmon": {
        "category": "Protein/Fish", "calories": 208, "protein": 28,
        "fat": 10, "carbs": 0, "glycemic_index": 0,
        "health_score": 10,
        "benefits": ["Omega-3 fatty acids", "High protein", "Heart healthy", "Anti-inflammatory"],
        "risks": ["Mercury (if eaten excessively)", "High calories"],
        "recommended_for": ["Heart disease", "Inflammation", "Brain health", "Athletes"],
        "avoid_if": ["Fish allergy", "Pregnancy (limit)"],
        "portion": "3-4 oz serving"
    },
    "white_bread": {
        "category": "Carbohydrate", "calories": 79, "protein": 2.7,
        "fat": 1, "carbs": 15, "glycemic_index": 75,
        "health_score": 3,
        "benefits": ["Quick energy"],
        "risks": ["High glycemic index", "Low fiber", "Nutrient poor", "Weight gain"],
        "recommended_for": ["Healthy individuals (occasional)"],
        "avoid_if": ["Diabetes", "Obesity", "Heart disease", "Celiac disease"],
        "portion": "1 slice"
    },
    "broccoli": {
        "category": "Vegetable", "calories": 55, "protein": 3.7,
        "fat": 0.6, "carbs": 11, "glycemic_index": 15,
        "health_score": 10,
        "benefits": ["High fiber", "Vitamin C", "Anti-cancer", "Low calorie"],
        "risks": ["Gas/bloating for sensitive individuals"],
        "recommended_for": ["Everyone", "Diabetes", "Cancer prevention", "Weight loss"],
        "avoid_if": ["Thyroid issues (limit raw)"],
        "portion": "1 cup"
    },
    "banana": {
        "category": "Fruit", "calories": 105, "protein": 1.3,
        "fat": 0.4, "carbs": 27, "glycemic_index": 51,
        "health_score": 7,
        "benefits": ["Potassium rich", "Quick energy", "Gut friendly"],
        "risks": ["High sugar content", "Moderate glycemic index"],
        "recommended_for": ["Athletes", "Heart health", "Pre-workout"],
        "avoid_if": ["Diabetes (limit)", "Kidney disease (high potassium)"],
        "portion": "1 medium"
    },
    "egg": {
        "category": "Protein", "calories": 78, "protein": 6,
        "fat": 5, "carbs": 0.6, "glycemic_index": 0,
        "health_score": 9,
        "benefits": ["Complete protein", "Vitamins B12/D", "Choline for brain"],
        "risks": ["Cholesterol (yolk)", "Saturated fat"],
        "recommended_for": ["Athletes", "Weight loss", "Brain health"],
        "avoid_if": ["Egg allergy", "High cholesterol (limit yolks)"],
        "portion": "1-2 eggs"
    },
    "soda": {
        "category": "Beverage", "calories": 150, "protein": 0,
        "fat": 0, "carbs": 39, "glycemic_index": 65,
        "health_score": 1,
        "benefits": ["None significant"],
        "risks": ["Very high sugar", "Obesity", "Diabetes", "Tooth decay", "Bone loss"],
        "recommended_for": ["None recommended"],
        "avoid_if": ["Diabetes", "Obesity", "Hypertension", "Heart disease"],
        "portion": "Avoid or minimize"
    },
    "oats": {
        "category": "Whole Grain", "calories": 166, "protein": 5.9,
        "fat": 3.6, "carbs": 28, "glycemic_index": 55,
        "health_score": 9,
        "benefits": ["Beta-glucan fiber", "Lowers cholesterol", "Slow energy release"],
        "risks": ["Gluten cross-contamination (some brands)"],
        "recommended_for": ["Heart disease", "Diabetes", "Weight management"],
        "avoid_if": ["Celiac disease", "Oat allergy"],
        "portion": "1/2 cup dry"
    },
    "avocado": {
        "category": "Healthy Fat", "calories": 234, "protein": 2.9,
        "fat": 21, "carbs": 12, "glycemic_index": 10,
        "health_score": 9,
        "benefits": ["Monounsaturated fats", "Potassium", "Fiber", "Heart healthy"],
        "risks": ["High calorie", "High fat (good fat but watch portions)"],
        "recommended_for": ["Heart disease", "Diabetes", "Brain health"],
        "avoid_if": ["Weight loss (moderate portions)", "Latex allergy"],
        "portion": "1/2 avocado"
    }
}


# ─────────────────────────────────────────────
# MEDICINE DATABASE
# ─────────────────────────────────────────────
MEDICINE_DB = {
    "metformin": {
        "class": "Biguanide", "type": "Antidiabetic",
        "used_for": ["Type 2 Diabetes", "PCOS", "Prediabetes"],
        "mechanism": "Reduces hepatic glucose production and improves insulin sensitivity",
        "common_side_effects": ["Nausea", "Diarrhea", "Stomach upset", "Lactic acidosis (rare)"],
        "serious_interactions": ["Alcohol", "Contrast dye", "Iodine supplements"],
        "take_with": "Food to reduce GI side effects",
        "monitor": ["Kidney function", "B12 levels", "Blood glucose"],
        "avoid_if": ["Kidney disease (CrCl <30)", "Liver disease", "Alcoholism"],
        "risk_level": "Low-Medium"
    },
    "aspirin": {
        "class": "NSAID/Antiplatelet", "type": "Analgesic/Antiplatelet",
        "used_for": ["Pain relief", "Fever", "Heart attack prevention", "Stroke prevention"],
        "mechanism": "Inhibits COX enzymes, reduces prostaglandins and thromboxane",
        "common_side_effects": ["GI bleeding", "Stomach irritation", "Ringing in ears"],
        "serious_interactions": ["Warfarin", "Other NSAIDs", "Alcohol"],
        "take_with": "Food or milk",
        "monitor": ["Signs of bleeding", "GI symptoms"],
        "avoid_if": ["Children under 16 (Reye's syndrome)", "Bleeding disorders", "Stomach ulcers"],
        "risk_level": "Medium"
    },
    "lisinopril": {
        "class": "ACE Inhibitor", "type": "Antihypertensive",
        "used_for": ["Hypertension", "Heart failure", "Post-MI", "Diabetic nephropathy"],
        "mechanism": "Inhibits ACE enzyme, reduces angiotensin II, lowers blood pressure",
        "common_side_effects": ["Dry cough", "Dizziness", "Headache", "Hyperkalemia"],
        "serious_interactions": ["Potassium supplements", "NSAIDs", "Lithium"],
        "take_with": "With or without food",
        "monitor": ["Blood pressure", "Potassium levels", "Kidney function"],
        "avoid_if": ["Pregnancy", "Angioedema history", "Severe kidney disease"],
        "risk_level": "Medium"
    },
    "atorvastatin": {
        "class": "Statin", "type": "Cholesterol-lowering",
        "used_for": ["High cholesterol", "Heart disease prevention", "Post-MI"],
        "mechanism": "Inhibits HMG-CoA reductase, reduces cholesterol synthesis in liver",
        "common_side_effects": ["Muscle pain", "Liver enzyme elevation", "Digestive issues"],
        "serious_interactions": ["Grapefruit juice", "Fibrates", "Niacin"],
        "take_with": "Any time of day, consistent timing",
        "monitor": ["Liver enzymes", "Creatine kinase", "Cholesterol levels"],
        "avoid_if": ["Pregnancy", "Active liver disease", "Unexplained muscle pain"],
        "risk_level": "Medium"
    },
    "ibuprofen": {
        "class": "NSAID", "type": "Analgesic/Anti-inflammatory",
        "used_for": ["Pain", "Fever", "Inflammation", "Arthritis"],
        "mechanism": "Non-selective COX inhibitor, reduces prostaglandins",
        "common_side_effects": ["Stomach upset", "GI bleeding", "Kidney stress"],
        "serious_interactions": ["Aspirin", "Warfarin", "ACE inhibitors", "Alcohol"],
        "take_with": "Food or milk to protect stomach",
        "monitor": ["Kidney function", "Blood pressure", "GI symptoms"],
        "avoid_if": ["Kidney disease", "Stomach ulcers", "Heart failure", "Third trimester pregnancy"],
        "risk_level": "Medium"
    },
    "amoxicillin": {
        "class": "Penicillin", "type": "Antibiotic",
        "used_for": ["Bacterial infections", "Ear infections", "Strep throat", "Pneumonia"],
        "mechanism": "Inhibits bacterial cell wall synthesis",
        "common_side_effects": ["Diarrhea", "Nausea", "Rash", "Yeast infection"],
        "serious_interactions": ["Warfarin", "Methotrexate", "Birth control (reduced efficacy)"],
        "take_with": "With or without food; complete full course",
        "monitor": ["Signs of allergic reaction", "Diarrhea (C. diff)", "Rash"],
        "avoid_if": ["Penicillin allergy", "Mononucleosis (rash risk)"],
        "risk_level": "Low-Medium"
    },
    "omeprazole": {
        "class": "Proton Pump Inhibitor", "type": "Antacid/Gastroprotective",
        "used_for": ["GERD", "Peptic ulcers", "H. pylori (combo)", "NSAID-induced gastritis"],
        "mechanism": "Irreversibly inhibits H+/K+ ATPase proton pump in stomach",
        "common_side_effects": ["Headache", "Diarrhea", "Nausea", "B12 deficiency (long-term)"],
        "serious_interactions": ["Clopidogrel", "Methotrexate", "Some antiretrovirals"],
        "take_with": "30-60 minutes before eating",
        "monitor": ["Magnesium levels (long-term)", "B12 levels", "Bone density"],
        "avoid_if": ["Osteoporosis (long-term use caution)", "Hypomagnesemia"],
        "risk_level": "Low"
    },
    "paracetamol": {
        "class": "Analgesic/Antipyretic", "type": "Pain & Fever Relief",
        "used_for": ["Mild-moderate pain", "Fever", "Headache", "Post-vaccination fever"],
        "mechanism": "Central COX inhibition; exact mechanism partly unclear",
        "common_side_effects": ["Generally well-tolerated", "Nausea (overdose)"],
        "serious_interactions": ["Alcohol (hepatotoxicity)", "Warfarin (high doses)"],
        "take_with": "With or without food",
        "monitor": ["Liver function (if high doses/chronic use)", "Total daily dose <4g"],
        "avoid_if": ["Liver disease", "Alcohol dependence", "G6PD deficiency"],
        "risk_level": "Low"
    }
}


# ─────────────────────────────────────────────
# DISEASE RISK RULES
# ─────────────────────────────────────────────
def compute_disease_risks(data):
    risks = {}

    # Diabetes
    db_score = 0
    if data.get('glucose', 0) >= 126:        db_score += 40
    elif data.get('glucose', 0) >= 100:      db_score += 20
    if data.get('bmi', 0) >= 30:             db_score += 20
    elif data.get('bmi', 0) >= 25:           db_score += 10
    if data.get('diabetic', 0):              db_score += 30
    if data.get('age', 0) >= 45:             db_score += 10
    risks['Diabetes'] = min(db_score, 95)

    # Hypertension
    ht_score = 0
    if data.get('systolic_bp', 0) >= 140:    ht_score += 40
    elif data.get('systolic_bp', 0) >= 130:  ht_score += 20
    if data.get('diastolic_bp', 0) >= 90:    ht_score += 20
    if data.get('bmi', 0) >= 30:             ht_score += 15
    if data.get('age', 0) >= 55:             ht_score += 15
    if data.get('smoker', 0):                ht_score += 10
    risks['Hypertension'] = min(ht_score, 95)

    # Cardiovascular
    cv_score = 0
    if data.get('cholesterol', 0) >= 240:    cv_score += 25
    elif data.get('cholesterol', 0) >= 200:  cv_score += 10
    if data.get('ldl', 0) >= 160:            cv_score += 25
    elif data.get('ldl', 0) >= 130:          cv_score += 10
    if data.get('hdl', 0) < 40:              cv_score += 20
    if data.get('smoker', 0):                cv_score += 20
    if data.get('age', 0) >= 55:             cv_score += 10
    risks['Cardiovascular Disease'] = min(cv_score, 95)

    # Anemia
    hb = data.get('hemoglobin', 15)
    an_score = 0
    if hb < 8:    an_score = 90
    elif hb < 10: an_score = 70
    elif hb < 12: an_score = 40
    elif hb < 13: an_score = 20
    risks['Anemia'] = an_score

    # Obesity
    bmi = data.get('bmi', 22)
    ob_score = 0
    if bmi >= 40:   ob_score = 95
    elif bmi >= 35: ob_score = 75
    elif bmi >= 30: ob_score = 55
    elif bmi >= 25: ob_score = 30
    risks['Obesity'] = ob_score

    return risks


# ─────────────────────────────────────────────
# RECOMMENDATIONS ENGINE
# ─────────────────────────────────────────────
def generate_recommendations(data, risk_level, disease_risks):
    recs = {
        "lifestyle": [],
        "diet": [],
        "exercise": [],
        "monitoring": [],
        "urgent": []
    }

    bmi      = data.get('bmi', 22)
    glucose  = data.get('glucose', 90)
    bp_sys   = data.get('systolic_bp', 120)
    chol     = data.get('cholesterol', 180)
    hb       = data.get('hemoglobin', 14)
    smoker   = data.get('smoker', 0)
    exercise = data.get('exercise_days', 3)

    if smoker:
        recs["urgent"].append("⚠️ Quit smoking immediately — it is the #1 modifiable risk factor for heart disease and cancer.")
    if glucose >= 126:
        recs["urgent"].append("⚠️ Fasting glucose ≥126 mg/dL — possible diabetes. Consult a doctor for HbA1c testing.")
    if bp_sys >= 180:
        recs["urgent"].append("⚠️ Systolic BP ≥180 — hypertensive crisis. Seek immediate medical attention.")
    if hb < 8:
        recs["urgent"].append("⚠️ Hemoglobin <8 g/dL — severe anemia. Medical evaluation required urgently.")

    if bmi > 25:
        recs["diet"].append("Reduce refined carbs and added sugars; choose whole grains.")
        recs["diet"].append("Increase fiber intake — vegetables, legumes, oats.")
    if glucose > 100:
        recs["diet"].append("Limit high-glycemic foods: white bread, soda, sweets.")
        recs["diet"].append("Eat smaller, more frequent meals to stabilize blood sugar.")
    if chol > 200:
        recs["diet"].append("Reduce saturated fats (red meat, full-fat dairy); increase omega-3s (salmon, flaxseed).")

    if exercise < 3:
        recs["exercise"].append("Target 150 min/week of moderate aerobic activity (brisk walking, cycling, swimming).")
        recs["exercise"].append("Add 2 days/week of strength training to boost metabolism.")
    else:
        recs["exercise"].append("Great activity level! Consider adding flexibility/yoga for holistic health.")

    if risk_level == "High":
        recs["monitoring"].append("Monthly blood pressure checks.")
        recs["monitoring"].append("Quarterly fasting glucose & HbA1c tests.")
        recs["monitoring"].append("Annual full lipid panel, kidney, and liver function tests.")
    elif risk_level == "Medium":
        recs["monitoring"].append("Blood pressure check every 3 months.")
        recs["monitoring"].append("Annual lipid panel and fasting glucose.")

    recs["lifestyle"].append("Aim for 7–9 hours of quality sleep per night.")
    recs["lifestyle"].append("Practice stress management: deep breathing, meditation, or yoga.")
    recs["lifestyle"].append("Stay hydrated — 8–10 glasses of water daily.")

    return recs


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "model_accuracy": f"{accuracy:.2%}",
        "version": "1.0.0"
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required = ['age', 'bmi', 'glucose', 'systolic_bp', 'diastolic_bp',
                    'cholesterol', 'hdl', 'ldl', 'hemoglobin', 'heart_rate',
                    'smoker', 'diabetic', 'exercise_days']

        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        features = [float(data[field]) for field in required]
        features_scaled = scaler.transform([features])

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        risk_labels = {0: "Low", 1: "Medium", 2: "High"}
        risk_level  = risk_labels[prediction]

        disease_risks = compute_disease_risks(data)
        recommendations = generate_recommendations(data, risk_level, disease_risks)

        bmi = data.get('bmi', 22)
        bmi_cat = ("Underweight" if bmi < 18.5 else
                   "Normal"      if bmi < 25   else
                   "Overweight"  if bmi < 30   else "Obese")

        return jsonify({
            "risk_level":      risk_level,
            "risk_score":      int(prediction),
            "confidence": {
                "low":    round(float(probabilities[0]) * 100, 1),
                "medium": round(float(probabilities[1]) * 100, 1),
                "high":   round(float(probabilities[2]) * 100, 1),
            },
            "disease_risks":    disease_risks,
            "recommendations":  recommendations,
            "bmi_category":     bmi_cat,
            "model_accuracy":  f"{accuracy:.2%}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/food', methods=['GET'])
def analyze_food():
    try:
        food_name = request.args.get('name', '').lower().replace(' ', '_')
        if not food_name:
            return jsonify({"error": "Provide ?name=foodname"}), 400

        # fuzzy match
        match = None
        for key in FOOD_DB:
            if food_name in key or key in food_name:
                match = key
                break

        if not match:
            available = list(FOOD_DB.keys())
            return jsonify({
                "error": f"Food '{food_name}' not found.",
                "available_foods": available
            }), 404

        food = FOOD_DB[match]
        return jsonify({"food": match, **food})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/medicine', methods=['GET'])
def analyze_medicine():
    try:
        med_name = request.args.get('name', '').lower().replace(' ', '_')
        if not med_name:
            return jsonify({"error": "Provide ?name=medicinename"}), 400

        match = None
        for key in MEDICINE_DB:
            if med_name in key or key in med_name:
                match = key
                break

        if not match:
            available = list(MEDICINE_DB.keys())
            return jsonify({
                "error": f"Medicine '{med_name}' not found.",
                "available_medicines": available
            }), 404

        med = MEDICINE_DB[match]
        return jsonify({"medicine": match, **med})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/foods', methods=['GET'])
def list_foods():
    return jsonify({"foods": list(FOOD_DB.keys())})


@app.route('/api/medicines', methods=['GET'])
def list_medicines():
    return jsonify({"medicines": list(MEDICINE_DB.keys())})


@app.route('/api/full_diagnosis', methods=['POST'])
def full_diagnosis():
    """Combined endpoint: health prediction + food + medicine analysis"""
    try:
        data      = request.get_json()
        vitals    = data.get('vitals', {})
        food_list = data.get('foods', [])
        med_list  = data.get('medicines', [])

        # Run base prediction
        required = ['age','bmi','glucose','systolic_bp','diastolic_bp',
                    'cholesterol','hdl','ldl','hemoglobin','heart_rate',
                    'smoker','diabetic','exercise_days']
        features  = [float(vitals.get(f, 0)) for f in required]
        features_scaled = scaler.transform([features])
        prediction      = model.predict(features_scaled)[0]
        probabilities   = model.predict_proba(features_scaled)[0]
        risk_labels     = {0: "Low", 1: "Medium", 2: "High"}
        risk_level      = risk_labels[prediction]
        disease_risks   = compute_disease_risks(vitals)
        recommendations = generate_recommendations(vitals, risk_level, disease_risks)

        # Food analysis
        food_results = {}
        for food in food_list:
            key = food.lower().replace(' ', '_')
            for db_key in FOOD_DB:
                if key in db_key or db_key in key:
                    food_results[food] = FOOD_DB[db_key]
                    break

        # Medicine analysis
        med_results = {}
        for med in med_list:
            key = med.lower()
            for db_key in MEDICINE_DB:
                if key in db_key or db_key in key:
                    med_results[med] = MEDICINE_DB[db_key]
                    break

        return jsonify({
            "health_prediction": {
                "risk_level": risk_level,
                "confidence": {
                    "low":    round(float(probabilities[0]) * 100, 1),
                    "medium": round(float(probabilities[1]) * 100, 1),
                    "high":   round(float(probabilities[2]) * 100, 1),
                },
                "disease_risks":   disease_risks,
                "recommendations": recommendations
            },
            "food_analysis":     food_results,
            "medicine_analysis": med_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 HealthScan ML Server starting on http://localhost:5000")
    app.run(debug=False, port=5000)