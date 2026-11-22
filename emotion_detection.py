"""Emotion detection module using IBM Watson NLU API.

This module provides the emotion_detector function for analyzing
emotions in text using the IBM Watson Natural Language Understanding service.
"""

import json
import os
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def emotion_detector(text: str) -> Optional[Dict]:
    """
    Detect emotions in the given text using Watson NLU API.

    Args:
        text: The text to analyze for emotions

    Returns:
        Dictionary with emotion scores and dominant emotion, or None if error
    """
    # Watson NLU API endpoint
    api_key = os.getenv('WATSON_NLU_API_KEY')
    url = os.getenv('WATSON_NLU_URL')

    if not api_key or not url:
        print("Error: Watson NLU credentials not configured")
        return None

    headers = {
        'Content-Type': 'application/json',
    }

    data = {
        'text': text,
        'features': {
            'emotion': {}
        }
    }

    try:
        response = requests.post(
            f'{url}/v1/analyze?version=2022-04-07',
            headers=headers,
            auth=('apikey', api_key),
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            emotions = result.get('emotion', {}).get('document', {}).get('emotion', {})

            # Find dominant emotion
            if emotions:
                dominant_emotion = max(emotions, key=emotions.get)
                return {
                    'anger': emotions.get('anger', 0),
                    'disgust': emotions.get('disgust', 0),
                    'fear': emotions.get('fear', 0),
                    'joy': emotions.get('joy', 0),
                    'sadness': emotions.get('sadness', 0),
                    'dominant_emotion': dominant_emotion
                }
            return None
        elif response.status_code == 400:
            print(f"Error: Bad request - invalid input text (status code 400)")
            return None
        else:
            print(f"Error: API returned status code {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error calling Watson NLU API: {e}")
        return None


if __name__ == '__main__':
    # Test the function
    test_text = "I love this wonderful day!"
    result = emotion_detector(test_text)
    print(f"Input: {test_text}")
    print(f"Result: {result}")