# source code created with AI assistance

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import EncodeRequestSerializer, DecodeRequestSerializer
from .sparsamp import encode_spar, decode_spar

class SparsampEncodeView(APIView):
    def post(self, request):
        serializer = EncodeRequestSerializer(data=request.data)
        if serializer.is_valid():
            encoded = encode_spar(
                serializer.validated_data['plaintext'],
                serializer.validated_data['key_material'],
                serializer.validated_data['random_seed']
            )
            return Response({"encoded_output": encoded})
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SparsampDecodeView(APIView):
    def post(self, request):
        serializer = DecodeRequestSerializer(data=request.data)
        if serializer.is_valid():
            decoded = decode_spar(
                serializer.validated_data['encoded_text'],
                serializer.validated_data['key_material'],
                serializer.validated_data['random_seed']
            )
            return Response({"decoded_output": decoded})
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
