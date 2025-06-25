# AI generated content

from rest_framework import serializers


class EncodeRequestSerializer(serializers.Serializer):
    plaintext = serializers.CharField()
    key_material = serializers.CharField()
    random_seed = serializers.IntegerField()


class DecodeRequestSerializer(serializers.Serializer):
    encoded_text = serializers.CharField()
    key_material = serializers.CharField()
    random_seed = serializers.IntegerField()
