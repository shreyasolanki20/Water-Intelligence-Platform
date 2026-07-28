import pandas as pd
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, accuracy_score,mean_absolute_error, mean_squared_error,classification_report, confusion_matrix

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv(
    "C:/Users/shreya/Downloads/Gujarat_Water_Intelligence_Dataset_100K (1).csv"
)

# ===================================================
# WATER CONSUMPTION MODEL (Regression)
# ===================================================

print("Training Water Consumption Model...")

X_demand = df[
    [
        "population",
        "rainfall_mm",
        "temperature_c",
        "humidity_percent",
        "reservoir_level_percent",
        "groundwater_level_m"
    ]
]

y_demand = df["water_consumption_mld"]

X_train, X_test, y_train, y_test = train_test_split(
    X_demand,
    y_demand,
    test_size=0.3,
    random_state=42
)

Demand_model = Ridge(alpha=1.0)

Demand_model.fit(X_train, y_train)

prediction = Demand_model.predict(X_test)

print("MAE :", mean_absolute_error(y_test, prediction))
print("MSE :", mean_squared_error(y_test, prediction))
print("R2 Score :", r2_score(y_test,prediction ))


joblib.dump(Demand_model, "models/demand.pkl", compress=3)


print("✅ demand.pkl Saved Successfully")


# ===================================================
# LEAKAGE DETECTION MODEL (Classification)
# ===================================================

print("\nTraining Leakage Detection Model...")

X_leakage = df[
    [
        "expected_flow_lpm",
        "actual_flow_lpm",
        "water_pressure_psi",
        "reservoir_outflow_mld"
    ]
]

y_leakage = df["leakage_detected"]

X_train, X_test, y_train, y_test = train_test_split(
    X_leakage,
    y_leakage,
    test_size=0.3,
    random_state=42
)

Leakage_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

Leakage_model.fit(X_train, y_train)

prediction = Leakage_model.predict(X_test)


print("Leakage Model Accuracy :", accuracy_score(y_test, prediction))
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test, prediction))

with open("models/leakage.pkl", "wb") as f:
    pickle.dump(Leakage_model, f)

print("✅ leakage.pkl Saved Successfully")

print("\n🎉 All Models Retrained Successfully!")