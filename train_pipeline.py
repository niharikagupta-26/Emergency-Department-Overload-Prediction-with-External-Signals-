"""
ER Overload Prediction — Pipeline v2 (Standalone Script)
Run this directly to train and inspect the model without the web server.
Output: er_daily_v2.csv + console metrics

Usage:
    python scripts/train_pipeline.py

Make sure Hospital_ER_Data.csv is in the data/ folder first.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'Hospital_ER_Data.csv')
OUT_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'er_daily_v2.csv')

# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 1 — Loading and cleaning the dataset")
print("=" * 55)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

df.drop(columns=["Patient Id","Patient First Inital","Patient Last Name","Patient Satisfaction Score"],
        inplace=True, errors="ignore")
df["Patient Admission Date"] = pd.to_datetime(df["Patient Admission Date"], dayfirst=True, errors="coerce")
df["Department Referral"]    = df["Department Referral"].fillna("Walk-in").astype(str)
df["Patient Admission Flag"] = df["Patient Admission Flag"].astype(int)
print(f"After cleaning: {len(df)} rows")

# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2 — Engineering features")
print("=" * 55)

df["hour"]       = df["Patient Admission Date"].dt.hour
df["weekday"]    = df["Patient Admission Date"].dt.dayofweek
df["month"]      = df["Patient Admission Date"].dt.month
df["date_only"]  = df["Patient Admission Date"].dt.date
df["is_weekend"] = (df["weekday"] >= 5).astype(int)
df["is_peak_hour"]= (((df["hour"]>=8)&(df["hour"]<=11))|((df["hour"]>=17)&(df["hour"]<=21))).astype(int)

le_dept = LabelEncoder()
df["dept_encoded"] = le_dept.fit_transform(df["Department Referral"])
print("Patient-level features created.")

# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3 — Aggregating to daily level")
print("=" * 55)

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
    ).reset_index()
)
daily.columns = ["date","total_visits","avg_wait_time","admission_rate",
                 "cm_flag_count","avg_age","weekend_visits","peak_hour_visits",
                 "walk_in_count","neurology_count","cardiology_count"]
daily["date"]        = pd.to_datetime(daily["date"])
daily["weekday"]     = daily["date"].dt.dayofweek
daily["month"]       = daily["date"].dt.month
daily["year"]        = daily["date"].dt.year
daily["quarter"]     = daily["date"].dt.quarter
daily["is_weekend"]  = (daily["weekday"] >= 5).astype(int)
daily["season_encoded"] = daily["month"].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
daily = daily.sort_values("date").reset_index(drop=True)
daily["lag_1_day"]       = daily["total_visits"].shift(1)
daily["lag_7_days"]      = daily["total_visits"].shift(7)
daily["rolling_7d_mean"] = daily["total_visits"].rolling(7).mean()
threshold = daily["total_visits"].quantile(0.75)
daily["is_overloaded"] = (daily["total_visits"] >= threshold).astype(int)
print(f"Daily rows: {len(daily)} | Overload threshold: ≥{threshold:.0f} visits/day")

# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4 — Weather & Event features")
print("=" * 55)

def sim_temp(row):
    base = {1:22,2:25,3:30,4:35,5:37,6:33,7:28,8:27,9:27,10:26,11:23,12:21}.get(row["month"],28)
    np.random.seed(int(str(row["date"])[:10].replace("-","")) % 10000)
    return round(base + np.random.uniform(-3, 3), 1)

def sim_hum(row):
    base = {1:55,2:50,3:40,4:35,5:40,6:70,7:80,8:82,9:75,10:65,11:60,12:57}.get(row["month"],55)
    np.random.seed((int(str(row["date"])[:10].replace("-","")) + 1) % 10000)
    return min(100, max(20, round(base + np.random.uniform(-8, 8))))

def sim_precip(row):
    prob = {1:0.05,2:0.05,3:0.05,4:0.07,5:0.12,6:0.55,7:0.70,8:0.65,9:0.55,10:0.25,11:0.10,12:0.05}.get(row["month"],0.1)
    np.random.seed((int(str(row["date"])[:10].replace("-","")) + 2) % 10000)
    return round(np.random.uniform(1,40),1) if np.random.random() < prob else 0.0

def weather_sev(temp, hum, precip):
    s=0
    if temp>=40: s+=4
    elif temp>=37: s+=2
    elif temp<=12: s+=3
    elif temp<=16: s+=1
    if hum>=80: s+=2
    elif hum>=70: s+=1
    if precip>=20: s+=3
    elif precip>=5: s+=1
    return min(10,s)

def event_risk(row):
    m,wd,d = row["month"], row["weekday"], row["date"].day
    r=0
    if m==10 and 5<=d<=25: r+=3
    if m==11 and 1<=d<=15: r+=2
    if m==1  and 12<=d<=16: r+=2
    if m==3  and 20<=d<=31: r+=2
    if m==4  and 1<=d<=10: r+=2
    if m in [4,5]: r+=1
    if wd==5: r+=1
    if wd==6: r+=2
    if d==1 and m in [1,4,8,10,11]: r+=1
    return min(5,r)

daily["temperature_c"]    = daily.apply(sim_temp, axis=1)
daily["humidity_pct"]     = daily.apply(sim_hum, axis=1)
daily["precipitation_mm"] = daily.apply(sim_precip, axis=1)
daily["weather_severity"] = daily.apply(lambda r: weather_sev(r["temperature_c"],r["humidity_pct"],r["precipitation_mm"]), axis=1)
daily["is_extreme_weather"]= (daily["weather_severity"]>=5).astype(int)
daily["is_heat_stress"]    = (daily["temperature_c"]>=38).astype(int)
daily["is_heavy_rain"]     = (daily["precipitation_mm"]>=10).astype(int)
daily["event_risk_score"]  = daily.apply(event_risk, axis=1)
daily["is_major_event"]    = (daily["event_risk_score"]>=3).astype(int)
daily["combined_stress"]   = (daily["weather_severity"] + daily["event_risk_score"]*1.5).round(1)

daily.dropna(inplace=True); daily.reset_index(drop=True, inplace=True)
print(f"Final rows: {len(daily)} | Extreme weather days: {daily['is_extreme_weather'].sum()}")

# ─────────────────────────────────────────────
FEATURES = [
    "weekday","month","year","quarter","is_weekend","season_encoded",
    "avg_wait_time","admission_rate","cm_flag_count","avg_age",
    "walk_in_count","peak_hour_visits","neurology_count","cardiology_count",
    "lag_1_day","lag_7_days","rolling_7d_mean",
    "temperature_c","humidity_pct","precipitation_mm",
    "weather_severity","is_extreme_weather","is_heat_stress","is_heavy_rain",
    "event_risk_score","is_major_event","combined_stress",
]

print("\n" + "=" * 55)
print("STEP 5 — Training Random Forest (200 trees)")
print("=" * 55)

X,y_class,y_reg = daily[FEATURES], daily["is_overloaded"], daily["total_visits"]
split = int(len(X)*0.80)
X_train,X_test   = X.iloc[:split], X.iloc[split:]
yc_train,yc_test = y_class.iloc[:split], y_class.iloc[split:]
yr_train,yr_test = y_reg.iloc[:split], y_reg.iloc[split:]
print(f"Train: {split} days | Test: {len(X)-split} days")

clf = RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_split=5,min_samples_leaf=2,random_state=42,class_weight="balanced")
clf.fit(X_train,yc_train); yc_pred=clf.predict(X_test)

reg = RandomForestRegressor(n_estimators=200,max_depth=10,min_samples_split=5,min_samples_leaf=2,random_state=42)
reg.fit(X_train,yr_train); yr_pred=reg.predict(X_test)

print("\n── CLASSIFIER ─────────────────────────────────────────")
print(classification_report(yc_test,yc_pred,target_names=["Normal","Overloaded"]))
cm = confusion_matrix(yc_test,yc_pred)
print(f"True Normal: {cm[0,0]} | False Alarm: {cm[0,1]}")
print(f"Missed:      {cm[1,0]} | Caught:      {cm[1,1]}")

mae = mean_absolute_error(yr_test, yr_pred)
r2  = r2_score(yr_test, yr_pred)
print(f"\n── REGRESSOR ─────────────────────────────────────────")
print(f"  MAE : {mae:.1f} patients/day")
print(f"  R²  : {r2:.3f}")

print("\n── FEATURE IMPORTANCE (top 10) ────────────────────────")
imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
NEW = ["temperature_c","humidity_pct","precipitation_mm","weather_severity",
       "is_extreme_weather","is_heat_stress","is_heavy_rain","event_risk_score","is_major_event","combined_stress"]
for f,v in imp.head(10).items():
    bar = "█" * int(v*40)
    tag = " ← NEW" if f in NEW else ""
    print(f"  {f:<28} {bar} {v:.3f}{tag}")

daily.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")
print("\n" + "=" * 55)
print("PIPELINE v2 COMPLETE")
print("=" * 55)
