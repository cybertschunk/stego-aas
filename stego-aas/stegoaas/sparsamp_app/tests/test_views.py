from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status


class SparsampViewsTest(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_encode_decode_roundtrip(self):
        """Test that encoding and then decoding returns the original plaintext."""
        context = "Once upon a time"
        text = "attack@dawn"
        seed = 12345
        encode_payload = {
            "plaintext": text,
            "context": context,
            "random_seed": seed
        }

        encode_response = self.client.post(
            "/sparsamp_app/api/encode/",
            encode_payload,
            format="json"
        )
        self.assertEqual(encode_response.status_code, status.HTTP_200_OK)
        self.assertIn("messages", encode_response.data)

        messages = encode_response.data["messages"]
        self.assertIsInstance(messages, list)
        self.assertGreater(len(messages), 0)

        decode_payload = {
            "messages": messages,
            "context": context,
            "random_seed": seed
        }

        decode_response = self.client.post(
            "/sparsamp_app/api/decode/",
            decode_payload,
            format="json"
        )
        self.assertEqual(decode_response.status_code, status.HTTP_200_OK)
        self.assertIn("message", decode_response.data)
        self.assertEqual(decode_response.data["message"], text)
        self.assertIn("attempts_per_message", decode_response.data)

    def test_encode_missing_fields(self):
        """Test that encode endpoint returns 400 when required fields are missing."""
        response = self.client.post(
            "/sparsamp_app/api/encode/",
            {"plaintext": "test"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_decode_missing_fields(self):
        """Test that decode endpoint returns 400 when required fields are missing."""
        response = self.client.post(
            "/sparsamp_app/api/decode/",
            {"messages": ["test"]},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
