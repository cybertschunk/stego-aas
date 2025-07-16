from django.apps import AppConfig


class SparSampAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sparsamp_app'

    def ready(self):
        from .sparsamp_utils import load_model
        load_model()

