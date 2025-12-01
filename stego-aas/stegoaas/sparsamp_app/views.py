# source code created with AI assistance

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import EncodeRequestSerializer, DecodeRequestSerializer, MessagesSerializer, MessageSerializer
from .encoding import full_encode
from .decoding import full_decode
from .sparsamp_utils import string_to_utf8_binary


class SparsampEncodeView(APIView):
    def post(self, request):
        serializer = EncodeRequestSerializer(data=request.data)
        if serializer.is_valid():
            requested_text = serializer.validated_data['plaintext']
            requested_context = serializer.validated_data['context']
            requested_seed = serializer.validated_data['random_seed']
            requested_text = string_to_utf8_binary(requested_text)
            messages = full_encode(requested_context, requested_text, requested_seed)
            serializer = MessagesSerializer({"messages": messages})
            return Response(serializer.data)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SparsampDecodeView(APIView):
    def post(self, request):
        serializer = DecodeRequestSerializer(data=request.data)
        if serializer.is_valid():
            messages = serializer.validated_data['messages']
            requested_context = serializer.validated_data['context']
            requested_seed = serializer.validated_data['random_seed']
            decoded_text, attempts_list = full_decode(context=requested_context, messages=messages, random_seed=requested_seed)
            serializer = MessageSerializer({"message": decoded_text})
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
