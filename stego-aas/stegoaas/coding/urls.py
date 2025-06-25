from django.urls import path

from . import views
from .views import SparsampEncodeView, SparsampDecodeView

urlpatterns = [
    path('api/encode/', SparsampEncodeView.as_view()),
    path('api/decode/', SparsampDecodeView.as_view()),
]