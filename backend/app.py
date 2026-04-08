"""
ER AutoPredict v2 — Flask Backend
Exposes the Random Forest pipeline as a REST API.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib, os, warnings, json
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
ROOT      = os.path.dirname(BASE)
DATA_DIR  = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
FRONT_DIR = os.path.join(ROOT, "frontend")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(FRONT_DIR, "static"),
            template_folder=os.path.join(FRONT_DIR, "templates"))
CORS(app)

# ── Global state ──────────────────────────────────────────────────
PIPELINE = {
    "clf":        None,
    "reg":        None,
    "daily":      None,
    "features":   None,
    "threshold":  None,
    "le_dept":    None,
    "metrics":    None,
    "trained_at": None,
}

# ═══════════════════════════════════════════════════════════════════
#  PIPELINE LOGIC  (mirrors er_auto_prediction_v2.py)
# ═══════════════════════════════════════════════════════════════════

def get_shift(hour):
    if 6  <= hour < 12: return "Morning"
    if 12 <= hour < 18: return "Afternoon"
    if 18 <= hour < 24: return "Evening"
    return "Night"

def get_season(month):
    if month in [12, 1, 2]:  return "Winter"
    if month in [3, 4, 5]:   return "Spring"
    if month in [6, 7, 8]:   return "Summer"
    return "Autumn"

def simulate_temperature(row):
    monthly_avg = {1:22,2:25,3:30,4:35,5:37,6:33,
                   7:28,8:27,9:27,10:26,11:23,12:21}
    base = monthly_avg.get(row["month"], 28)
    np.random.seed(int(str(row["date"])[:10].replace("-","")) % 10000)
    return round(base + np.random.uniform(-3, 3), 1)

def simulate_humidity(row):
    monthly_avg = {1:55,2:50,3:40,4:35,5:40,6:70,
                   7:80,8:82,9:75,10:65,11:60,12:57}
    base = monthly_avg.get(row["month"], 55)
    np.random.seed((int(str(row["date"])[:10].replace("-","")) + 1) % 10000)
    return min(100, max(20, round(base + np.random.uniform(-8, 8))))

def simulate_precipitation(row):
    monthly_prob = {1:0.05,2:0.05,3:0.05,4:0.07,5:0.12,6:0.55,
                    7:0.70,8:0.65,9:0.55,10:0.25,11:0.10,12:0.05}
    prob = monthly_prob.get(row["month"], 0.1)
    np.random.seed((int(str(row["date"])[:10].replace("-","")) + 2) % 10000)
    if np.random.random() < prob:
        return round(np.random.uniform(1, 40), 1)
    return 0.0

def get_weather_severity(temp, humidity, precip):
    score = 0
    if temp >= 40:        score += 4
    elif temp >= 37:      score += 2
    elif temp <= 12:      score += 3
    elif temp <= 16:      score += 1
    if humidity >= 80:    score += 2
    elif humidity >= 70:  score += 1
    if precip >= 20:      score += 3
    elif precip >= 5:     score += 1
    return min(10, score)

def get_event_risk(row):
    month   = row["month"]
    weekday = row["weekday"]
    day     = row["date"].day
    risk = 0
    if month == 10 and 5  <= day <= 25: risk += 3
    if month == 11 and 1  <= day <= 15: risk += 2
    if month == 1  and 12 <= day <= 16: risk += 2
    if month == 3  and 20 <= day <= 31: risk += 2
    if month == 4  and 1  <= day <= 10: risk += 2
    if month in [4, 5]:                 risk += 1
    if weekday == 5:  risk += 1
    if weekday == 6:  risk += 2
    if day == 1 and month in [1,4,8,10,11]: risk += 1
    return min(5, risk)

FEATURES = [
    "weekday","month","year","quarter","is_weekend","season_encoded",
    "avg_wait_time","admission_rate","cm_flag_count",
    "avg_age","walk_in_count","peak_hour_visits",
    "neurology_count","cardiology_count",
    "lag_1_day","lag_7_days","rolling_7d_mean",
    "temperature_c","humidity_pct","precipitation_mm",
    "weather_severity","is_extreme_weather","is_heat_stress","is_heavy_rain",
    "event_risk_score","is_major_event",
    "combined_stress",
]

def run_pipeline(csv_path: str):
    """Full 5-step pipeline. Returns trained models + daily dataframe."""
    # STEP 1 — Load & clean
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.drop(columns=["Patient Id","Patient First Inital",
                      "Patient Last Name","Patient Satisfaction Score"],
            inplace=True, errors="ignore")
    df["Patient Admission Date"] = pd.to_datetime(
        df["Patient Admission Date"], dayfirst=True, errors="coerce")
    df["Department Referral"]    = df["Department Referral"].fillna("Walk-in").astype(str)
    df["Patient Admission Flag"] = df["Patient Admission Flag"].astype(int)

    # STEP 2 — Feature engineering
    df["hour"]        = df["Patient Admission Date"].dt.hour
    df["weekday"]     = df["Patient Admission Date"].dt.dayofweek
    df["month"]       = df["Patient Admission Date"].dt.month
    df["date_only"]   = df["Patient Admission Date"].dt.date
    df["is_weekend"]  = (df["weekday"] >= 5).astype(int)
    df["is_peak_hour"]= (((df["hour"]>=8)&(df["hour"]<=11))|
                         ((df["hour"]>=17)&(df["hour"]<=21))).astype(int)

    le_dept = LabelEncoder()
    df["dept_encoded"] = le_dept.fit_transform(df["Department Referral"])

    # STEP 3 — Aggregate daily
    daily = (
        df.groupby("date_only")
        .agg(
            total_visits     =("hour","count"),
            avg_wait_time    =("Patient Waittime","mean"),
            admission_rate   =("Patient Admission Flag","mean"),
            cm_flag_count    =("Patients CM","sum"),
            avg_age          =("Patient Age","mean"),
            weekend_visits   =("is_weekend","sum"),
            peak_hour_visits =("is_peak_hour","sum"),
            walk_in_count    =("Department Referral", lambda x:(x=="Walk-in").sum()),
            neurology_count  =("Department Referral", lambda x:(x=="Neurology").sum()),
            cardiology_count =("Department Referral", lambda x:(x=="Cardiology").sum()),
        )
        .reset_index()
    )
    daily.columns = ["date","total_visits","avg_wait_time","admission_rate",
                     "cm_flag_count","avg_age","weekend_visits","peak_hour_visits",
                     "walk_in_count","neurology_count","cardiology_count"]

    daily["date"]           = pd.to_datetime(daily["date"])
    daily["weekday"]        = daily["date"].dt.dayofweek
    daily["month"]          = daily["date"].dt.month
    daily["year"]           = daily["date"].dt.year
    daily["quarter"]        = daily["date"].dt.quarter
    daily["is_weekend"]     = (daily["weekday"] >= 5).astype(int)
    daily["season_encoded"] = daily["month"].map(
        {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["lag_1_day"]       = daily["total_visits"].shift(1)
    daily["lag_7_days"]      = daily["total_visits"].shift(7)
    daily["rolling_7d_mean"] = daily["total_visits"].rolling(7).mean()

    threshold = daily["total_visits"].quantile(0.75)
    daily["is_overloaded"] = (daily["total_visits"] >= threshold).astype(int)

    # STEP 4 — Weather & events
    daily["temperature_c"]    = daily.apply(simulate_temperature, axis=1)
    daily["humidity_pct"]     = daily.apply(simulate_humidity, axis=1)
    daily["precipitation_mm"] = daily.apply(simulate_precipitation, axis=1)
    daily["weather_severity"] = daily.apply(
        lambda r: get_weather_severity(r["temperature_c"], r["humidity_pct"], r["precipitation_mm"]), axis=1)
    daily["is_extreme_weather"]= (daily["weather_severity"] >= 5).astype(int)
    daily["is_heat_stress"]    = (daily["temperature_c"] >= 38).astype(int)
    daily["is_heavy_rain"]     = (daily["precipitation_mm"] >= 10).astype(int)
    daily["event_risk_score"]  = daily.apply(get_event_risk, axis=1)
    daily["is_major_event"]    = (daily["event_risk_score"] >= 3).astype(int)
    daily["combined_stress"]   = (daily["weather_severity"] + daily["event_risk_score"]*1.5).round(1)

    daily.dropna(inplace=True)
    daily.reset_index(drop=True, inplace=True)

    # STEP 5 — Train
    X       = daily[FEATURES]
    y_class = daily["is_overloaded"]
    y_reg   = daily["total_visits"]

    split_idx = int(len(X) * 0.80)
    X_train, X_test   = X.iloc[:split_idx], X.iloc[split_idx:]
    yc_train, yc_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]
    yr_train, yr_test = y_reg.iloc[:split_idx],   y_reg.iloc[split_idx:]

    clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  min_samples_split=5, min_samples_leaf=2,
                                  random_state=42, class_weight="balanced")
    clf.fit(X_train, yc_train)
    yc_pred = clf.predict(X_test)
    yc_prob = clf.predict_proba(X_test)[:,1]

    reg = RandomForestRegressor(n_estimators=200, max_depth=10,
                                 min_samples_split=5, min_samples_leaf=2,
                                 random_state=42)
    reg.fit(X_train, yr_train)
    yr_pred = reg.predict(X_test)

    mae = mean_absolute_error(yr_test, yr_pred)
    r2  = r2_score(yr_test, yr_pred)
    cr  = classification_report(yc_test, yc_pred, target_names=["Normal","Overloaded"], output_dict=True)
    cm  = confusion_matrix(yc_test, yc_pred).tolist()

    importances = dict(zip(FEATURES, clf.feature_importances_.round(4).tolist()))

    # Persist models
    joblib.dump(clf, os.path.join(MODEL_DIR,"clf.pkl"))
    joblib.dump(reg, os.path.join(MODEL_DIR,"reg.pkl"))
    joblib.dump(le_dept, os.path.join(MODEL_DIR,"le_dept.pkl"))
    daily.to_csv(os.path.join(DATA_DIR,"er_daily_v2.csv"), index=False)

    return clf, reg, daily, threshold, le_dept, {
        "mae": round(mae, 2),
        "r2":  round(r2, 3),
        "classification_report": cr,
        "confusion_matrix": cm,
        "feature_importances": importances,
        "train_days": int(split_idx),
        "test_days":  int(len(X) - split_idx),
        "total_days": int(len(daily)),
        "overload_threshold": int(threshold),
        "test_start": str(daily["date"].iloc[split_idx].date()),
        "test_end":   str(daily["date"].iloc[-1].date()),
    }

# ── Load or train at startup ───────────────────────────────────────
def load_or_train():
    clf_path  = os.path.join(MODEL_DIR, "clf.pkl")
    reg_path  = os.path.join(MODEL_DIR, "reg.pkl")
    dept_path = os.path.join(MODEL_DIR, "le_dept.pkl")
    data_path = os.path.join(DATA_DIR, "er_daily_v2.csv")
    csv_path  = os.path.join(DATA_DIR, "Hospital_ER_Data.csv")

    if all(os.path.exists(p) for p in [clf_path, reg_path, dept_path, data_path]):
        PIPELINE["clf"]    = joblib.load(clf_path)
        PIPELINE["reg"]    = joblib.load(reg_path)
        PIPELINE["le_dept"]= joblib.load(dept_path)
        PIPELINE["daily"]  = pd.read_csv(data_path, parse_dates=["date"])
        PIPELINE["features"]  = FEATURES
        PIPELINE["threshold"] = PIPELINE["daily"]["total_visits"].quantile(0.75)
        imp = dict(zip(FEATURES, PIPELINE["clf"].feature_importances_.round(4).tolist()))
        PIPELINE["metrics"] = {"feature_importances": imp, "cached": True}
        PIPELINE["trained_at"] = "cached"
        print("[ER] Loaded cached models.")
    elif os.path.exists(csv_path):
        print("[ER] Training from CSV …")
        clf, reg, daily, thr, le, metrics = run_pipeline(csv_path)
        PIPELINE.update(clf=clf, reg=reg, daily=daily, features=FEATURES,
                        threshold=thr, le_dept=le, metrics=metrics,
                        trained_at=datetime.now().isoformat())
        print("[ER] Training complete.")
    else:
        print("[ER] No CSV found. Upload Hospital_ER_Data.csv via /api/upload")

load_or_train()

# ═══════════════════════════════════════════════════════════════════
#  PREDICTION HELPER
# ═══════════════════════════════════════════════════════════════════

def build_feature_row(
    date_str=None, hour=None,
    temperature_c=None, humidity_pct=None, precipitation_mm=None,
    avg_wait_time=None, admission_rate=None, avg_age=None,
    walk_in_count=None, peak_hour_visits=None,
    cm_flag_count=None, neurology_count=None, cardiology_count=None,
):
    """Build a single feature row for live prediction."""
    daily = PIPELINE["daily"]
    if daily is None:
        return None, "Model not trained"

    target_date = pd.to_datetime(date_str) if date_str else pd.Timestamp.now()
    hour = hour if hour is not None else datetime.now().hour

    # Weather (use provided or simulate)
    mock_row = {"date": target_date, "month": target_date.month}
    temp  = temperature_c    if temperature_c    is not None else simulate_temperature(mock_row)
    hum   = humidity_pct     if humidity_pct     is not None else simulate_humidity(mock_row)
    prec  = precipitation_mm if precipitation_mm is not None else simulate_precipitation(mock_row)
    wsev  = get_weather_severity(temp, hum, prec)

    # Events
    evt_row = {"date": target_date, "month": target_date.month,
               "weekday": target_date.dayofweek}
    evt_risk = get_event_risk(evt_row)

    # Lag features from daily history
    lag1 = daily["total_visits"].iloc[-1] if len(daily) >= 1 else 50
    lag7 = daily["total_visits"].iloc[-7] if len(daily) >= 7 else 50
    roll = daily["total_visits"].tail(7).mean()

    season_map = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}

    row = {
        "weekday":           target_date.dayofweek,
        "month":             target_date.month,
        "year":              target_date.year,
        "quarter":           target_date.quarter,
        "is_weekend":        int(target_date.dayofweek >= 5),
        "season_encoded":    season_map.get(target_date.month, 2),
        "avg_wait_time":     avg_wait_time    if avg_wait_time    is not None else daily["avg_wait_time"].tail(7).mean(),
        "admission_rate":    admission_rate   if admission_rate   is not None else daily["admission_rate"].tail(7).mean(),
        "cm_flag_count":     cm_flag_count    if cm_flag_count    is not None else daily["cm_flag_count"].tail(7).mean(),
        "avg_age":           avg_age          if avg_age          is not None else daily["avg_age"].mean(),
        "walk_in_count":     walk_in_count    if walk_in_count    is not None else daily["walk_in_count"].tail(7).mean(),
        "peak_hour_visits":  peak_hour_visits if peak_hour_visits is not None else daily["peak_hour_visits"].tail(7).mean(),
        "neurology_count":   neurology_count  if neurology_count  is not None else daily["neurology_count"].tail(7).mean(),
        "cardiology_count":  cardiology_count if cardiology_count is not None else daily["cardiology_count"].tail(7).mean(),
        "lag_1_day":         lag1,
        "lag_7_days":        lag7,
        "rolling_7d_mean":   roll,
        "temperature_c":     temp,
        "humidity_pct":      hum,
        "precipitation_mm":  prec,
        "weather_severity":  wsev,
        "is_extreme_weather":int(wsev >= 5),
        "is_heat_stress":    int(temp >= 38),
        "is_heavy_rain":     int(prec >= 10),
        "event_risk_score":  evt_risk,
        "is_major_event":    int(evt_risk >= 3),
        "combined_stress":   round(wsev + evt_risk * 1.5, 1),
    }
    return pd.DataFrame([row])[FEATURES], None

# ═══════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(FRONT_DIR, "index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(FRONT_DIR, "static"), filename)


@app.route("/api/status")
def api_status():
    trained = PIPELINE["clf"] is not None
    return jsonify({
        "trained":    trained,
        "trained_at": PIPELINE.get("trained_at"),
        "total_days": int(len(PIPELINE["daily"])) if trained else 0,
        "threshold":  int(PIPELINE["threshold"])  if trained else None,
    })


@app.route("/api/train", methods=["POST"])
def api_train():
    csv_path = os.path.join(DATA_DIR, "Hospital_ER_Data.csv")
    if not os.path.exists(csv_path):
        return jsonify({"error": "Hospital_ER_Data.csv not found in data/"}), 404
    try:
        clf, reg, daily, thr, le, metrics = run_pipeline(csv_path)
        PIPELINE.update(clf=clf, reg=reg, daily=daily, features=FEATURES,
                        threshold=thr, le_dept=le, metrics=metrics,
                        trained_at=datetime.now().isoformat())
        return jsonify({"ok": True, "metrics": metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Must be a CSV"}), 400
    save_path = os.path.join(DATA_DIR, "Hospital_ER_Data.csv")
    f.save(save_path)
    # Auto-train after upload
    try:
        clf, reg, daily, thr, le, metrics = run_pipeline(save_path)
        PIPELINE.update(clf=clf, reg=reg, daily=daily, features=FEATURES,
                        threshold=thr, le_dept=le, metrics=metrics,
                        trained_at=datetime.now().isoformat())
        return jsonify({"ok": True, "rows": int(len(daily)), "metrics": metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["GET","POST"])
def api_predict():
    if PIPELINE["clf"] is None:
        return jsonify({"error": "Model not trained. Upload CSV first."}), 503

    params = request.json or {}
    X_row, err = build_feature_row(**params)
    if err:
        return jsonify({"error": err}), 400

    clf, reg = PIPELINE["clf"], PIPELINE["reg"]
    predicted_class = int(clf.predict(X_row)[0])
    overload_prob   = float(clf.predict_proba(X_row)[0][1])
    predicted_visits= int(round(reg.predict(X_row)[0]))

    risk_score = int(round(overload_prob * 100))
    risk_level = "HIGH" if overload_prob >= 0.65 else "MEDIUM" if overload_prob >= 0.40 else "LOW"

    row = X_row.iloc[0].to_dict()

    return jsonify({
        "predicted_overload":  predicted_class,
        "overload_probability": round(overload_prob, 3),
        "predicted_visits":    predicted_visits,
        "risk_score":          risk_score,
        "risk_level":          risk_level,
        "features_used": {
            "temperature_c":    row["temperature_c"],
            "humidity_pct":     row["humidity_pct"],
            "precipitation_mm": row["precipitation_mm"],
            "weather_severity": row["weather_severity"],
            "event_risk_score": row["event_risk_score"],
            "combined_stress":  row["combined_stress"],
            "is_weekend":       int(row["is_weekend"]),
            "lag_1_day":        int(row["lag_1_day"]),
            "rolling_7d_mean":  round(row["rolling_7d_mean"], 1),
        }
    })


@app.route("/api/forecast", methods=["GET"])
def api_forecast():
    """7-day ahead forecast."""
    if PIPELINE["clf"] is None:
        return jsonify({"error": "Model not trained"}), 503
    results = []
    today = datetime.now()
    for delta in range(7):
        target = today + timedelta(days=delta)
        X_row, _ = build_feature_row(date_str=str(target.date()))
        if X_row is None:
            continue
        prob   = float(PIPELINE["clf"].predict_proba(X_row)[0][1])
        visits = int(round(PIPELINE["reg"].predict(X_row)[0]))
        risk   = "HIGH" if prob >= 0.65 else "MEDIUM" if prob >= 0.40 else "LOW"
        row    = X_row.iloc[0]
        results.append({
            "date":          str(target.date()),
            "day":           target.strftime("%a %d"),
            "predicted_visits":    visits,
            "overload_probability": round(prob, 3),
            "risk_level":    risk,
            "risk_score":    int(round(prob * 100)),
            "temperature_c": row["temperature_c"],
            "event_risk":    int(row["event_risk_score"]),
            "weather_severity": int(row["weather_severity"]),
        })
    return jsonify(results)


@app.route("/api/metrics")
def api_metrics():
    if PIPELINE["metrics"] is None:
        return jsonify({"error": "Not trained"}), 503
    return jsonify(PIPELINE["metrics"])


@app.route("/api/history")
def api_history():
    if PIPELINE["daily"] is None:
        return jsonify([])
    daily = PIPELINE["daily"]
    tail  = daily.tail(60).copy()
    tail["date"] = tail["date"].dt.strftime("%Y-%m-%d")
    cols = ["date","total_visits","avg_wait_time","admission_rate",
            "temperature_c","humidity_pct","precipitation_mm",
            "weather_severity","event_risk_score","combined_stress",
            "is_overloaded","is_extreme_weather","is_major_event"]
    return jsonify(tail[cols].to_dict(orient="records"))


@app.route("/api/feature_importance")
def api_feature_importance():
    if PIPELINE["clf"] is None:
        return jsonify({"error": "Not trained"}), 503
    imp = sorted(
        zip(FEATURES, PIPELINE["clf"].feature_importances_),
        key=lambda x: -x[1]
    )
    return jsonify([{"feature": f, "importance": round(v, 4)} for f, v in imp])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
