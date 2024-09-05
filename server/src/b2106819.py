from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Load the trained model and Iris dataset
model = joblib.load('svm_model.pkl')
label_names = ['setosa', 'versicolor', 'virginica']

def predict_iris_label(petal_length, petal_width, sepal_length, sepal_width):
    features = [[petal_length, petal_width, sepal_length, sepal_width]]
    prediction = model.predict(features)
    predicted_label = prediction[0]
    predicted_class = label_names[predicted_label]
    return predicted_label, predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])

            predicted_label, predicted_class = predict_iris_label(petal_length, petal_width, sepal_length, sepal_width)
            result_label = predicted_label
            result_class = predicted_class
            result = f"Predicted label: {result_label}, Predicted class: {result_class}"
        except ValueError:
            result_label = result_class = None
            result = "Please enter valid numerical values."

        return render_template('index.html', result=result, result_label=result_label, result_class=result_class)

    return render_template('index.html', result=None, result_label=None, result_class=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

