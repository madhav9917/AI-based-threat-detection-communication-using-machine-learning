from flask import Flask, request, jsonify
from src.predict import predict

app = Flask(__name__)

@app.route('/')
def home():
    return "IDS Running"

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json['features']
    result = predict(data)

    if result == "normal":
        return jsonify({"result": "Normal traffic"})
    else:
        return jsonify({"result": f"⚠️ Threat Detected: {result.upper()}"})

if __name__ == "__main__":
    app.run(debug=True)
