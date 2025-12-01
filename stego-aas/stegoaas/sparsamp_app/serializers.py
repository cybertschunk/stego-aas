# AI generated content

from rest_framework import serializers


class EncodeRequestSerializer(serializers.Serializer):
    plaintext = serializers.CharField(trim_whitespace=False)
    context = serializers.CharField(trim_whitespace=False)
    random_seed = serializers.IntegerField()


class DecodeRequestSerializer(serializers.Serializer):
    messages = serializers.ListField(
        child=serializers.CharField(trim_whitespace=False)
    )
    context = serializers.CharField(trim_whitespace=False)
    random_seed = serializers.IntegerField()

class MessagesSerializer(serializers.Serializer):
    messages = serializers.ListField(
        child=serializers.CharField(trim_whitespace=False)
    )

