from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib, numpy as np

app = Flask(__name__)
CORS(app)  

modelo = joblib.load('fruta_modelo.pkl')
scaler = joblib.load('fruta_scaler.pkl')

# serve o HTML
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/prever', methods=['POST'])
def prever():
    dados = request.json
    entrada = np.array([[dados['peso'], dados['tamanho']]])
    entrada_n = scaler.transform(entrada)
    pred = modelo.predict(entrada_n)[0]
    return jsonify({'resultado': pred})

if __name__ == '__main__':
    app.run(debug=True)
