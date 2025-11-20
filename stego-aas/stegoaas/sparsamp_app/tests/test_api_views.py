"""
Test suite for SparSamp API views.

Tests the encode and decode REST API endpoints with various inputs
including valid requests, invalid data, and error cases.
"""

from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient


class SparsampEncodeViewTest(TestCase):
    """Test cases for the /api/encode/ endpoint."""

    def setUp(self):
        """Set up test client and common test data."""
        self.client = APIClient()
        self.url = '/sparsamp_app/api/encode/'
        self.valid_payload = {
            'plaintext': 'attack@dawn',
            'context': 'Once upon a time',
            'random_seed': 12345
        }

    def test_encode_valid_request(self):
        """Test encoding with valid data returns 200 and messages."""
        response = self.client.post(self.url, self.valid_payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('messages', response.data)
        self.assertIsInstance(response.data['messages'], list)
        self.assertGreater(len(response.data['messages']), 0)
        # Each message should be a non-empty string
        for message in response.data['messages']:
            self.assertIsInstance(message, str)
            self.assertGreater(len(message), 0)

    def test_encode_empty_plaintext(self):
        """Test encoding with empty plaintext returns 400."""
        payload = self.valid_payload.copy()
        payload['plaintext'] = ''
        response = self.client.post(self.url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('plaintext', response.data)

    def test_encode_missing_plaintext(self):
        """Test encoding without plaintext field returns 400."""
        payload = self.valid_payload.copy()
        del payload['plaintext']
        response = self.client.post(self.url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('plaintext', response.data)

    def test_encode_empty_context(self):
        """Test encoding with empty context returns 400."""
        payload = self.valid_payload.copy()
        payload['context'] = ''
        response = self.client.post(self.url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('context', response.data)

    def test_encode_missing_context(self):
        """Test encoding without context field returns 400."""
        payload = self.valid_payload.copy()
        del payload['context']
        response = self.client.post(self.url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('context', response.data)

    def test_encode_invalid_random_seed_type(self):
        """Test encoding with non-integer random_seed returns 400."""
        payload = self.valid_payload.copy()
        payload['random_seed'] = 'not_an_integer'
        response = self.client.post(self.url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('random_seed', response.data)

    def test_encode_missing_random_seed(self):
        """Test encoding without random_seed field returns 400."""
        payload = self.valid_payload.copy()
        del payload['random_seed']
        response = self.client.post(self.url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('random_seed', response.data)

    def test_encode_long_plaintext(self):
        """Test encoding with longer plaintext still works."""
        payload = self.valid_payload.copy()
        payload['plaintext'] = "This is a much longer message that should still be encoded successfully by the system."
        response = self.client.post(self.url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('messages', response.data)
        self.assertGreater(len(response.data['messages']), 0)

    def test_encode_different_context(self):
        """Test encoding with different context strings."""
        payload = self.valid_payload.copy()
        payload['context'] = 'In a galaxy far far away'
        response = self.client.post(self.url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('messages', response.data)

    def test_encode_get_method_not_allowed(self):
        """Test that GET requests to encode endpoint return 405."""
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)


class SparsampDecodeViewTest(TestCase):
    """Test cases for the /api/decode/ endpoint."""

    def setUp(self):
        """Set up test client and encode a message for decoding tests."""
        self.client = APIClient()
        self.encode_url = '/sparsamp_app/api/encode/'
        self.decode_url = '/sparsamp_app/api/decode/'

        # Encode a message first to get valid encoded messages
        self.plaintext = 'attack@dawn'
        self.context = 'Once upon a time'
        self.random_seed = 12345

        encode_payload = {
            'plaintext': self.plaintext,
            'context': self.context,
            'random_seed': self.random_seed
        }

        encode_response = self.client.post(self.encode_url, encode_payload, format='json')
        self.encoded_messages = encode_response.data['messages']

    def test_decode_valid_request(self):
        """Test decoding with valid encoded messages returns original plaintext."""
        payload = {
            'messages': self.encoded_messages,
            'context': self.context,
            'random_seed': self.random_seed
        }
        response = self.client.post(self.decode_url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('message', response.data)
        self.assertEqual(response.data['message'], self.plaintext)

    def test_decode_empty_messages_list(self):
        """Test decoding with empty messages list returns 400."""
        payload = {
            'messages': [],
            'context': self.context,
            'random_seed': self.random_seed
        }
        response = self.client.post(self.decode_url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_decode_missing_messages(self):
        """Test decoding without messages field returns 400."""
        payload = {
            'context': self.context,
            'random_seed': self.random_seed
        }
        response = self.client.post(self.decode_url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('messages', response.data)

    def test_decode_invalid_messages_type(self):
        """Test decoding with non-list messages returns 400."""
        payload = {
            'messages': 'not_a_list',
            'context': self.context,
            'random_seed': self.random_seed
        }
        response = self.client.post(self.decode_url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('messages', response.data)

    def test_decode_empty_context(self):
        """Test decoding with empty context returns 400."""
        payload = {
            'messages': self.encoded_messages,
            'context': '',
            'random_seed': self.random_seed
        }
        response = self.client.post(self.decode_url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('context', response.data)

    def test_decode_missing_context(self):
        """Test decoding without context field returns 400."""
        payload = {
            'messages': self.encoded_messages,
            'random_seed': self.random_seed
        }
        response = self.client.post(self.decode_url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('context', response.data)

    def test_decode_invalid_random_seed_type(self):
        """Test decoding with non-integer random_seed returns 400."""
        payload = {
            'messages': self.encoded_messages,
            'context': self.context,
            'random_seed': 'not_an_integer'
        }
        response = self.client.post(self.decode_url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('random_seed', response.data)

    def test_decode_missing_random_seed(self):
        """Test decoding without random_seed field returns 400."""
        payload = {
            'messages': self.encoded_messages,
            'context': self.context
        }
        response = self.client.post(self.decode_url, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('random_seed', response.data)

    def test_decode_wrong_context(self):
        """Test decoding with wrong context fails with 500 error."""
        payload = {
            'messages': self.encoded_messages,
            'context': 'Different context string',
            'random_seed': self.random_seed
        }
        response = self.client.post(self.decode_url, payload, format='json')

        # Should return 500 error since decoding will fail
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn('error', response.data)

    def test_decode_wrong_random_seed(self):
        """Test decoding with wrong random_seed fails with 500 error."""
        payload = {
            'messages': self.encoded_messages,
            'context': self.context,
            'random_seed': 99999  # Different seed
        }
        response = self.client.post(self.decode_url, payload, format='json')

        # Should return 500 error since decoding will fail
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn('error', response.data)

    def test_decode_get_method_not_allowed(self):
        """Test that GET requests to decode endpoint return 405."""
        response = self.client.get(self.decode_url)

        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)


class SparsampEndToEndTest(TestCase):
    """End-to-end integration tests for encode and decode API workflow."""

    def setUp(self):
        """Set up test client."""
        self.client = APIClient()
        self.encode_url = '/sparsamp_app/api/encode/'
        self.decode_url = '/sparsamp_app/api/decode/'

    def test_full_encode_decode_cycle(self):
        """Test complete encode-decode cycle maintains data integrity."""
        plaintext = 'Secret message'
        context = 'The sky was blue'
        random_seed = 54321

        # Encode
        encode_payload = {
            'plaintext': plaintext,
            'context': context,
            'random_seed': random_seed
        }
        encode_response = self.client.post(self.encode_url, encode_payload, format='json')
        self.assertEqual(encode_response.status_code, status.HTTP_200_OK)

        # Decode
        decode_payload = {
            'messages': encode_response.data['messages'],
            'context': context,
            'random_seed': random_seed
        }
        decode_response = self.client.post(self.decode_url, decode_payload, format='json')
        self.assertEqual(decode_response.status_code, status.HTTP_200_OK)

        # Verify integrity
        self.assertEqual(decode_response.data['message'], plaintext)

    def test_multiple_messages_with_different_seeds(self):
        """Test encoding multiple plaintexts with different seeds."""
        # Use tested parameters that work reliably
        test_cases = [
            {
                'plaintext': 'attack@dawn',
                'context': 'Once upon a time',
                'random_seed': 12345
            },
            {
                'plaintext': 'Secret message',
                'context': 'The sky was blue',
                'random_seed': 54321
            },
            {
                'plaintext': 'Hidden communication',
                'context': 'In a distant land',
                'random_seed': 98765
            }
        ]

        for test_case in test_cases:
            # Encode
            encode_response = self.client.post(self.encode_url, test_case, format='json')
            self.assertEqual(encode_response.status_code, status.HTTP_200_OK)
            self.assertIn('messages', encode_response.data)

            # Decode
            decode_payload = {
                'messages': encode_response.data['messages'],
                'context': test_case['context'],
                'random_seed': test_case['random_seed']
            }
            decode_response = self.client.post(self.decode_url, decode_payload, format='json')
            self.assertEqual(decode_response.status_code, status.HTTP_200_OK)
            self.assertEqual(decode_response.data['message'], test_case['plaintext'])
