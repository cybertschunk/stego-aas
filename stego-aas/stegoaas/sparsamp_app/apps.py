from django.apps import AppConfig


class SparSampAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sparsamp_app'

    def ready(self):
        from .model_manager import get_model_manager
        # Load the model when Django starts
        model_manager = get_model_manager()
        model_manager.load()

