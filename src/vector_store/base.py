"""
Abstract base interface for vector stores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Mapping, Sequence

import numpy as np


class VectorStore(ABC):
    """
    Minimal interface for storing embeddings.

    This scaffold focuses on ingestion (write) only. Search/query methods can be
    added later as needed.
    """

    @abstractmethod
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        texts: Sequence[str],
        metadatas: Iterable[Mapping[str, object]] | None = None,
    ) -> None:
        """
        Persist the embeddings and associated texts/metadata.
        """


