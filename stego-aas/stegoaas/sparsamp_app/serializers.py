# AI generated content

from rest_framework import serializers


class EncodeRequestSerializer(serializers.Serializer):
    plaintext = serializers.CharField()
    context = serializers.CharField()
    random_seed = serializers.IntegerField()


class DecodeRequestSerializer(serializers.Serializer):
    messages = serializers.ListField(
        child=serializers.CharField()
    )
    context = serializers.CharField()
    random_seed = serializers.IntegerField()

class MessagesSerializer(serializers.Serializer):
    messages = serializers.ListField(
        child=serializers.CharField()
    )

class MessageSerializer(serializers.Serializer):
    message = serializers.CharField()