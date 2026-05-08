import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

#1. cargar Dataset (ignorrla primera fila si pusimos encabezados)
df = pd.read_csv("Jupyter/Tesis-Proyecto/data/dataset.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:,-1].values

#2. Estandarización (relevante para el entrenamiento ML)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42,stratify= y
    )

#4. Entrenamiento con validacion cruzada
#Usamos balance de estimadores para no saturar a la raspberry

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

#Validamos que elmodelo sea estable
scores = cross_val_score(model,X_train,y_train,cv=5)
print(f"Presicion media de entrenamiento: {scores.mean():.2f}")

model.fit(X_train, y_train)

#5. Evaluacion final
y_pred = model.predict(X_test)
print(f"\n === Reporte de clasificacion ===\n",
      classification_report(y_test,y_pred))

#6. Guardar todo (MODELO + ESCALADOR)
joblib.dump(model, "Jupyter/Tesis-Proyecto/data/model.pkl")
joblib.dump(scaler,"Jupyter/Tesis-Proyecto/data/scaler.pkl")

print("\n Modelo y Escalador guardados correctamente")
'''
 === Reporte de clasificacion ===
               precision    recall  f1-score   support

           0       0.89      1.00      0.94        31
           1       1.00      0.85      0.92        26

    accuracy                           0.93        57
   macro avg       0.94      0.92      0.93        57
weighted avg       0.94      0.93      0.93        57


 Modelo y Escalador guardados correctamente
'''