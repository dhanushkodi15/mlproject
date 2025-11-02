from flask import Flask, request, render_template, jsonify
from src.components.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET'])
def prediction_page():
    return render_template('prediction.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        # Map frontend keys to model's expected column names
        data_dict = {
            "gender": data.get("gender"),
            "race/ethnicity": data.get("race_ethnicity"),
            "parental level of education": data.get("parental_level_of_education"),
            "lunch": data.get("lunch"),
            "test preparation course": data.get("test_preparation_course"),
            "reading score": float(data.get("reading_score")),
            "writing score": float(data.get("writing_score"))
        }
        import pandas as pd
        data_df = pd.DataFrame([data_dict])
        result = PredictPipeline().predict(data_df)
        return jsonify({'success': True, 'predicted_score': result[0]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)