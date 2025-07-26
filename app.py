import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and helpers
model = joblib.load('C:\\Users\\Dell\\OneDrive\\Desktop\\intership project\\iris classification\\model\\iris_model.pkl')
scaler = joblib.load('C:\\Users\\Dell\\OneDrive\\Desktop\\intership project\\iris classification\\model\\iris_scaler.pkl')
encoder = joblib.load('C:\\Users\\Dell\\OneDrive\\Desktop\\intership project\\iris classification\\model\\iris_label_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get input and predict
            input_values = [float(x) for x in request.form.values()]
            final_input = scaler.transform([input_values])
            prediction = model.predict(final_input)
            label = encoder.inverse_transform(prediction)[0]
            return render_template('index.html', prediction_text=f"The predicted species is: {label}")
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}")
    
    # For GET request, show blank form
    return render_template('index.html', prediction_text=None)

if __name__ == '__main__':
    app.run(debug=True)
