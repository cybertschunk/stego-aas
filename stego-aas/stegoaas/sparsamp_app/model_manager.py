"""
Singleton manager for GPT-2 model and tokenizer.

This module provides a thread-safe singleton pattern for managing the language model,
tokenizer, and device used throughout the steganography application.
"""

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .constants import MODEL_NAME

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton class for managing the GPT-2 model and tokenizer.

    Ensures only one instance of the model and tokenizer exist in memory,
    and provides thread-safe access to these resources.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the model manager (only runs once)."""
        # Prevent re-initialization
        if ModelManager._initialized:
            return

        self._model = None
        self._tokenizer = None
        self._device = None
        ModelManager._initialized = True

    def load(self):
        """
        Load the GPT-2 model and tokenizer.

        This method is idempotent - calling it multiple times will not
        reload the model if it's already loaded.
        """
        if self._model is not None:
            return  # Already loaded

        self._model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Log device information
        if torch.cuda.is_available():
            logger.info(f"Loading model on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available. Loading model on CPU.")

        self._model.to(self._device)
        self._model.eval()
        logger.info(f"Model loaded successfully on device: {self._device}")

    @property
    def model(self):
        """
        Get the loaded model.

        Returns:
            The GPT-2 model instance

        Raises:
            RuntimeError: If the model hasn't been loaded yet
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self):
        """
        Get the loaded tokenizer.

        Returns:
            The GPT-2 tokenizer instance

        Raises:
            RuntimeError: If the tokenizer hasn't been loaded yet
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer

    @property
    def device(self):
        """
        Get the device (CPU/GPU) used for model inference.

        Returns:
            torch.device instance

        Raises:
            RuntimeError: If the device hasn't been set yet
        """
        if self._device is None:
            raise RuntimeError("Device not set. Call load() first.")
        return self._device

    def is_loaded(self):
        """
        Check if the model and tokenizer are loaded.

        Returns:
            True if loaded, False otherwise
        """
        return self._model is not None and self._tokenizer is not None


# Convenience function to get the singleton instance
def get_model_manager():
    """
    Get the ModelManager singleton instance.

    Returns:
        ModelManager instance
    """
    return ModelManager()
