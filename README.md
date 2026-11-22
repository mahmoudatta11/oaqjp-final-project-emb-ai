# Final Project

A Python application that detects emotions in text using IBM Watson Natural Language Understanding (NLU) API, with a Flask web interface for easy testing.

## Project Structure

```
emotion_detection_project/
├── EmotionDetection/
│   ├── __init__.py          # Package initialization
│   └── emotion_detection.py # Core emotion detection logic
├── server.py                # Flask web server
├── test_emotion_detection.py # Unit tests
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## Features

- Detects 5 emotions: anger, disgust, fear, joy, sadness
- Identifies dominant emotion
- RESTful API endpoint for emotion analysis
- Web interface for interactive testing
- Comprehensive unit tests
- Error handling for blank input and API errors

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd emotion_detection_project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Watson NLU credentials:**
   ```bash
   cp .env.example .env
   # Edit .env with your IBM Watson NLU API key and URL
   ```

4. **Get Watson NLU credentials:**
   - Create an IBM Cloud account
   - Create a Natural Language Understanding service
   - Copy the API key and service URL to `.env`

## Usage

### Run the Web Server
```bash
python server.py
```
Then open http://localhost:5000 in your browser.

### Use as a Module
```python
from EmotionDetection.emotion_detection import emotion_detector

result = emotion_detector("I love this wonderful day!")
print(result)
# Output: {'anger': 0.01, 'disgust': 0.0, 'fear': 0.02, 'joy': 0.95, 'sadness': 0.02, 'dominant_emotion': 'joy'}
```

### Run Tests
```bash
python -m pytest test_emotion_detection.py -v
# or
python test_emotion_detection.py
```

### Run Static Code Analysis
```bash
pylint EmotionDetection/emotion_detection.py server.py
```

## API Endpoints

- `GET /` - Web interface
- `POST /analyze` - Analyze emotion in text
  - Request: `{"text": "your text here"}`
  - Response: `{"anger": 0.1, "disgust": 0.05, "fear": 0.15, "joy": 0.6, "sadness": 0.1, "dominant_emotion": "joy"}`

## Error Handling

- Blank input returns 400 status with error message
- API failures return 500 status with error message
- Missing credentials handled gracefully

## License

MIT License
