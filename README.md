# ER AutoPredict — Full-Stack Web Application

A production-ready web dashboard powered by a Random Forest ML pipeline,
integrating real-time weather (Open-Meteo API) and local event signals
to predict Hyderabad ER overload risk.

---

## 📁 Project Structure

```
er_autopredict/
├── backend/
│   └── app.py              ← Flask REST API + ML pipeline
├── frontend/
│   └── index.html          ← Dashboard UI (integrates with backend)
├── data/
│   └── Hospital_ER_Data.csv  ← Drop your CSV here
├── models/
│   ├── clf.pkl             
│   ├── reg.pkl             
│   └── le_dept.pkl         
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Python dependencies

```bash
cd er_autopredict
pip install -r requirements.txt
```

### 2. Add your data

Copy `Hospital_ER_Data.csv` into the `data/` folder:

```bash
cp /path/to/Hospital_ER_Data.csv data/
```

**Required columns:**
- `Patient Id`, `Patient First Inital`, `Patient Last Name`
- `Patient Admission Date` (DD/MM/YYYY HH:MM format)
- `Patient Age`, `Patient Gender` (M/F/NC)
- `Patient Waittime` (minutes)
- `Patient Admission Flag` (0/1)
- `Department Referral` (e.g. Neurology, Cardiology, Walk-in)
- `Patients CM`, `Patient Satisfaction Score`

### 3. Start the Flask backend

```bash
python backend/app.py
```

The server starts on `http://localhost:5000`.  
The pipeline auto-trains on startup if `Hospital_ER_Data.csv` is present.

### 4. Open the dashboard

Open your browser at: **http://localhost:5000**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/status` | Model training status |
| POST | `/api/train` | Re-train from existing CSV |
| POST | `/api/upload` | Upload new CSV + auto-train |
| POST | `/api/predict` | Get live RF prediction |
| GET | `/api/forecast` | 7-day ahead forecast |
| GET | `/api/metrics` | MAE, R², classification report |
| GET | `/api/history` | Last 60 days of daily aggregates |
| GET | `/api/feature_importance` | All 27 feature importances |

### `/api/predict` — POST body (all optional, uses smart defaults)

```json
{
  "temperature_c": 36.5,
  "humidity_pct": 55,
  "precipitation_mm": 0.0,
  "avg_wait_time": 35.2,
  "admission_rate": 0.45,
  "avg_age": 42.0
}
```

### `/api/predict` — Response

```json
{
  "predicted_overload": 1,
  "overload_probability": 0.73,
  "predicted_visits": 187,
  "risk_score": 73,
  "risk_level": "HIGH",
  "features_used": {
    "temperature_c": 36.5,
    "humidity_pct": 55,
    "weather_severity": 3,
    "event_risk_score": 2,
    "combined_stress": 6.0,
    ...
  }
}
```

---

## 🧠 ML Pipeline (v2)

**Step 1 — Load & Clean**: Drop PII columns, parse datetime, impute nulls.

**Step 2 — Feature Engineering** (patient-level):
- Time: hour, day, weekday, month, quarter, shift, is_peak_hour
- Demographics: age_group, gender_encoded
- Department: label-encoded referral type
- Season: Winter/Spring/Summer/Autumn

**Step 3 — Daily Aggregation**:
- Aggregate all patient rows by date
- Compute: total_visits, avg_wait_time, admission_rate, walk_in_count, etc.
- Lag features: lag_1_day, lag_7_days, rolling_7d_mean

**Step 4 — Weather & Event Features** ★ NEW:
- `temperature_c`, `humidity_pct`, `precipitation_mm`
- `weather_severity` (0–10 ER-impact score)
- `is_heat_stress`, `is_heavy_rain`, `is_extreme_weather`
- `event_risk_score` (0–5 Hyderabad event calendar)
- `is_major_event`, `combined_stress`

**Step 5 — Random Forest** (200 trees, depth 10):
- **Classifier**: Overloaded (yes/no) — trained with balanced class weights
- **Regressor**: Predicted patient count
- 80/20 time-based train/test split

**27 total features** across time, behaviour, weather, and events.

---

## 🌤️ Weather Data

Live weather fetched from **Open-Meteo** (no API key required):
- Latitude: 17.385°N (Hyderabad)
- Longitude: 78.487°E
- Variables: temperature, feels_like, humidity, wind_speed, precipitation, weather_code
- Falls back to historically-realistic simulated values if API is unavailable

---

## 🎪 Event Detection

Client-side calendar scan for Hyderabad-specific ER-relevant events:
- Festivals: Dussehra, Diwali, Sankranti, Holi, Ugadi
- Sports: IPL Cricket (Apr–May), ICC Tournaments (Sep–Oct)
- Seasonal: Monsoon (Jul–Sep), Summer peak (Apr–Jun), Marathons (Feb)
- Weekly: Friday/Saturday nightlife surge patterns

---

## 🚀 Deployment Notes

For production deployment:

```bash
# Use gunicorn instead of Flask dev server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```

For cloud deployment (AWS/GCP/Azure):
- The `models/` folder persists trained models — mount as a volume
- The `data/` folder holds the CSV — use object storage for large files
- CORS is enabled for all origins — restrict to your domain in production

---

## 📊 Dashboard Features

- **Live weather panel**: Real Open-Meteo API data for Hyderabad
- **Event scanner**: Automatic detection of 12+ local event categories  
- **Risk gauge**: Visual 0–100 overload risk score
- **7-day RF forecast**: Backend-powered, model-driven forecast chart
- **Feature importance**: Live from trained RF model (with NEW labels)
- **Model metrics**: MAE, R², Precision, Recall, F1-Score
- **60-day history**: Trend chart with overload threshold line
- **Prediction log**: Last 8 predictions with timestamps
- **Smart recommendations**: Contextual staff/supply suggestions per risk level
- **CSV upload**: Drag-drop upload + auto-retrain from browser
