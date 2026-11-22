"""Unit tests for emotion detection module."""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """Test cases for emotion detection function."""

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_emotion_detector_happy_path(self, mock_post):
        """Test emotion detector with successful API response."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'emotion': {
                'document': {
                    'emotion': {
                        'anger': 0.1,
                        'disgust': 0.05,
                        'fear': 0.15,
                        'joy': 0.6,
                        'sadness': 0.1
                    }
                }
            }
        }
        mock_post.return_value = mock_response

        # Set environment variables
        with patch.dict(os.environ, {
            'WATSON_NLU_API_KEY': 'test_key',
            'WATSON_NLU_URL': 'https://api.test.com'
        }):
            result = emotion_detector("I am so happy today!")

        self.assertIsNotNone(result)
        self.assertEqual(result['anger'], 0.1)
        self.assertEqual(result['disgust'], 0.05)
        self.assertEqual(result['fear'], 0.15)
        self.assertEqual(result['joy'], 0.6)
        self.assertEqual(result['sadness'], 0.1)
        self.assertEqual(result['dominant_emotion'], 'joy')

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_emotion_detector_all_emotions(self, mock_post):
        """Test emotion detector returns all five emotions."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'emotion': {
                'document': {
                    'emotion': {
                        'anger': 0.2,
                        'disgust': 0.1,
                        'fear': 0.1,
                        'joy': 0.3,
                        'sadness': 0.3
                    }
                }
            }
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {
            'WATSON_NLU_API_KEY': 'test_key',
            'WATSON_NLU_URL': 'https://api.test.com'
        }):
            result = emotion_detector("Test text")

        self.assertIn('anger', result)
        self.assertIn('disgust', result)
        self.assertIn('fear', result)
        self.assertIn('joy', result)
        self.assertIn('sadness', result)
        self.assertIn('dominant_emotion', result)

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_emotion_detector_dominant_emotion(self, mock_post):
        """Test that dominant emotion is correctly identified."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'emotion': {
                'document': {
                    'emotion': {
                        'anger': 0.1,
                        'disgust': 0.05,
                        'fear': 0.05,
                        'joy': 0.1,
                        'sadness': 0.7
                    }
                }
            }
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {
            'WATSON_NLU_API_KEY': 'test_key',
            'WATSON_NLU_URL': 'https://api.test.com'
        }):
            result = emotion_detector("I feel very sad")

        self.assertEqual(result['dominant_emotion'], 'sadness')

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_emotion_detector_api_error(self, mock_post):
        """Test emotion detector handles API error."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {'error': 'Bad request'}
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {
            'WATSON_NLU_API_KEY': 'test_key',
            'WATSON_NLU_URL': 'https://api.test.com'
        }):
            result = emotion_detector("Test text")

        self.assertIsNone(result)

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_emotion_detector_network_error(self, mock_post):
        """Test emotion detector handles network error."""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")

        with patch.dict(os.environ, {
            'WATSON_NLU_API_KEY': 'test_key',
            'WATSON_NLU_URL': 'https://api.test.com'
        }):
            result = emotion_detector("Test text")

        self.assertIsNone(result)

    def test_emotion_detector_missing_credentials(self):
        """Test emotion detector handles missing credentials."""
        with patch.dict(os.environ, {}, clear=True):
            result = emotion_detector("Test text")

        self.assertIsNone(result)


class TestEmotionDetectionFormat(unittest.TestCase):
    """Test cases for emotion detection output format."""

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_output_format_keys(self, mock_post):
        """Test that output has all required keys."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'emotion': {
                'document': {
                    'emotion': {
                        'anger': 0.1,
                        'disgust': 0.05,
                        'fear': 0.15,
                        'joy': 0.6,
                        'sadness': 0.1
                    }
                }
            }
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {
            'WATSON_NLU_API_KEY': 'test_key',
            'WATSON_NLU_URL': 'https://api.test.com'
        }):
            result = emotion_detector("Test text")

        expected_keys = {'anger', 'disgust', 'fear', 'joy', 'sadness', 'dominant_emotion'}
        self.assertEqual(set(result.keys()), expected_keys)

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_output_values_are_floats(self, mock_post):
        """Test that emotion values are floats."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'emotion': {
                'document': {
                    'emotion': {
                        'anger': 0.1,
                        'disgust': 0.05,
                        'fear': 0.15,
                        'joy': 0.6,
                        'sadness': 0.1
                    }
                }
            }
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {
            'WATSON_NLU_API_KEY': 'test_key',
            'WATSON_NLU_URL': 'https://api.test.com'
        }):
            result = emotion_detector("Test text")

        for key in ['anger', 'disgust', 'fear', 'joy', 'sadness']:
            self.assertIsInstance(result[key], float)

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_dominant_emotion_is_string(self, mock_post):
        """Test that dominant_emotion is a string."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'emotion': {
                'document': {
                    'emotion': {
                        'anger': 0.1,
                        'disgust': 0.05,
                        'fear': 0.15,
                        'joy': 0.6,
                        'sadness': 0.1
                    }
                }
            }
        }
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {
            'WATSON_NLU_API_KEY': 'test_key',
            'WATSON_NLU_URL': 'https://api.test.com'
        }):
            result = emotion_detector("Test text")

        self.assertIsInstance(result['dominant_emotion'], str)


if __name__ == '__main__':
    unittest.main()