from flask import Flask, request, jsonify
import joblib  # or pickle, or your preferred model loading

app = Flask(__name__)

# Load the trained model
model = joblib.load('old_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
