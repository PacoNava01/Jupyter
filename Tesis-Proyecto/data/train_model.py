import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# CARGAR DATASET
# -----------------------------
data = np.loadtxt("Jupyter/Tesis-Proyecto/data/dataset.csv", delimiter=",")

X = data[:, :-1]  # features
y = data[:, -1]   # labels

print(f"Total muestras: {len(X)}")

# -----------------------------
# SPLIT TRAIN / TEST
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# ENTRENAR MODELO
# -----------------------------
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUAR
# -----------------------------
y_pred = model.predict(X_test)

print("\n=== REPORTE ===")
print(classification_report(y_test, y_pred))

print("\n=== MATRIZ DE CONFUSIÓN ===")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# GUARDAR MODELO
# -----------------------------
joblib.dump(model, "Jupyter/Tesis-Proyecto/data/model.pkl")

print("\n[✔] Modelo guardado en data/model.pkl")