"""Flask server for emotion detection web application.

This module provides a Flask web application for detecting emotions
in text using the IBM Watson NLU API.
"""

from flask import Flask, request, jsonify, render_template_string
from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Detection App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 100px;
            box-sizing: border-box;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
            display: none;
        }
        .result.show {
            display: block;
        }
        .emotion-bar {
            margin: 10px 0;
        }
        .emotion-label {
            display: inline-block;
            width: 100px;
            font-weight: bold;
        }
        .emotion-bar-container {
            display: inline-block;
            width: 60%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }
        .emotion-bar-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        .anger { background-color: #dc3545; }
        .disgust { background-color: #6f42c1; }
        .fear { background-color: #fd7e14; }
        .joy { background-color: #28a745; }
        .sadness { background-color: #007bff; }
        .dominant {
            margin-top: 20px;
            padding: 15px;
            background-color: #e9ecef;
            border-radius: 5px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .error {
            color: #dc3545;
            padding: 15px;
            background-color: #f8d7da;
            border-radius: 5px;
            margin-top: 20px;
            display: none;
        }
        .error.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Emotion Detection App</h1>
        <p style="text-align: center; color: #666;">Analyze emotions in text using IBM Watson NLU</p>

        <div class="form-group">
            <label for="textInput">Enter text to analyze:</label>
            <textarea id="textInput" placeholder="Type or paste text here..."></textarea>
        </div>

        <button onclick="analyzeEmotion()">Analyze Emotion</button>

        <div class="error" id="errorDiv"></div>

        <div class="result" id="resultDiv">
            <h3>Emotion Analysis Results</h3>
            <div class="emotion-bar">
                <span class="emotion-label">Anger:</span>
                <div class="emotion-bar-container">
                    <div class="emotion-bar-fill anger" id="angerBar" style="width: 0%"></div>
                </div>
                <span id="angerValue">0%</span>
            </div>
            <div class="emotion-bar">
                <span class="emotion-label">Disgust:</span>
                <div class="emotion-bar-container">
                    <div class="emotion-bar-fill disgust" id="disgustBar" style="width: 0%"></div>
                </div>
                <span id="disgustValue">0%</span>
            </div>
            <div class="emotion-bar">
                <span class="emotion-label">Fear:</span>
                <div class="emotion-bar-container">
                    <div class="emotion-bar-fill fear" id="fearBar" style="width: 0%"></div>
                </div>
                <span id="fearValue">0%</span>
            </div>
            <div class="emotion-bar">
                <span class="emotion-label">Joy:</span>
                <div class="emotion-bar-container">
                    <div class="emotion-bar-fill joy" id="joyBar" style="width: 0%"></div>
                </div>
                <span id="joyValue">0%</span>
            </div>
            <div class="emotion-bar">
                <span class="emotion-label">Sadness:</span>
                <div class="emotion-bar-container">
                    <div class="emotion-bar-fill sadness" id="sadnessBar" style="width: 0%"></div>
                </div>
                <span id="sadnessValue">0%</span>
            </div>
            <div class="dominant" id="dominantDiv">
                Dominant Emotion: <span id="dominantEmotion">-</span>
            </div>
        </div>
    </div>

    <script>
        function analyzeEmotion() {
            const text = document.getElementById('textInput').value.trim();
            const errorDiv = document.getElementById('errorDiv');
            const resultDiv = document.getElementById('resultDiv');

            errorDiv.classList.remove('show');
            resultDiv.classList.remove('show');

            if (!text) {
                errorDiv.textContent = 'Error: Please enter some text to analyze.';
                errorDiv.classList.add('show');
                return;
            }

            fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    errorDiv.textContent = data.error;
                    errorDiv.classList.add('show');
                } else {
                    displayResults(data);
                }
            })
            .catch(err => {
                errorDiv.textContent = 'Error: Failed to analyze emotion. Please try again.';
                errorDiv.classList.add('show');
            });
        }

        function displayResults(data) {
            const resultDiv = document.getElementById('resultDiv');

            document.getElementById('angerBar').style.width = (data.anger * 100) + '%';
            document.getElementById('angerValue').textContent = (data.anger * 100).toFixed(1) + '%';

            document.getElementById('disgustBar').style.width = (data.disgust * 100) + '%';
            document.getElementById('disgustValue').textContent = (data.disgust * 100).toFixed(1) + '%';

            document.getElementById('fearBar').style.width = (data.fear * 100) + '%';
            document.getElementById('fearValue').textContent = (data.fear * 100).toFixed(1) + '%';

            document.getElementById('joyBar').style.width = (data.joy * 100) + '%';
            document.getElementById('joyValue').textContent = (data.joy * 100).toFixed(1) + '%';

            document.getElementById('sadnessBar').style.width = (data.sadness * 100) + '%';
            document.getElementById('sadnessValue').textContent = (data.sadness * 100).toFixed(1) + '%';

            document.getElementById('dominantEmotion').textContent = data.dominant_emotion.charAt(0).toUpperCase() + data.dominant_emotion.slice(1);

            resultDiv.classList.add('show');
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the main page with the emotion detection web interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze emotion in the provided text via POST request.

    Expects JSON with 'text' field. Returns emotion scores and dominant emotion.
    """
    data = request.get_json()

    # Handle missing text field
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text'].strip()

    # Handle blank/whitespace-only input
    if not text:
        return jsonify({'error': 'Text cannot be blank'}), 400

    result = emotion_detector(text)

    if result is None:
        return jsonify({'error': 'Failed to analyze emotion. Please check your Watson NLU configuration.'}), 500

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)