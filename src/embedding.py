"""
Embedding utilities wrapping Sentence Transformers.
"""

from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

from .logging_utils import get_logger


logger = get_logger(__name__)


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer with lazy loading.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model '%s'...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Model loaded.")
        return self._model

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list: List[str] = list(texts)
        logger.info("Encoding %d texts into embeddings", len(texts_list))
        embeddings = self.model.encode(
            texts_list,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return embeddings



