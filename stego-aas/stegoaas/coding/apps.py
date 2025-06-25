from django.apps import AppConfig


class CodingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'coding'

    def ready(self):
        from .utils import load_model
        load_model()

