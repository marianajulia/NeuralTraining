import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("apples_and_oranges.csv")
X, y = df[['Weight','Size']], df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)

print("Acurácia:", accuracy_score(y_test, model.predict(X_test_s)))

joblib.dump(model,  "fruta_modelo.pkl")
joblib.dump(scaler, "fruta_scaler.pkl")
