# """
# Embedding utilities wrapping Sentence Transformers with support for faster inference backends.
# """

# from typing import Iterable, List

# import numpy as np
# from sentence_transformers import SentenceTransformer

# from .logging_utils import get_logger


# logger = get_logger(__name__)


# class EmbeddingModel:
#     """
#     Thin wrapper around SentenceTransformer with lazy loading and optimized inference options.
    
#     Supports:
#     - Standard PyTorch inference (default)
#     - ONNX Runtime for faster inference (2-4x speedup)
#     - Configurable batch size for better throughput
#     - Device selection (CPU/GPU)
#     """

#     def __init__(
#         self,
#         model_name: str,
#         use_onnx: bool = False,
#         batch_size: int = 32,
#         device: str | None = None,
#     ) -> None:
#         """
#         Initialize embedding model.

#         Args:
#             model_name: HuggingFace model name (e.g., 'intfloat/multilingual-e5-large')
#             use_onnx: If True, use ONNX Runtime for faster inference (requires onnxruntime)
#             batch_size: Batch size for encoding (larger = faster but more memory)
#             device: Device to use ('cpu', 'cuda', or None for auto-detection)
#         """
#         self.model_name = model_name
#         self.use_onnx = use_onnx
#         self.batch_size = batch_size
#         self.device = device
#         self._model: SentenceTransformer | None = None

#     @property
#     def model(self) -> SentenceTransformer:
#         if self._model is None:
#             logger.info("Loading embedding model '%s'...", self.model_name)
            
#             # Load model with device specification if provided
#             model_kwargs = {}
#             if self.device:
#                 model_kwargs["device"] = self.device
            
#             self._model = SentenceTransformer(self.model_name, **model_kwargs)
            
#             # Convert to ONNX if requested
#             if self.use_onnx:
#                 try:
#                     logger.info("ONNX mode enabled - model will use ONNX Runtime when available")
#                     # sentence-transformers will automatically use ONNX if available
#                     # The conversion happens on first encode if the ONNX model exists
#                 except ImportError:
#                     logger.warning(
#                         "ONNX Runtime not available. Install with: pip install onnxruntime"
#                     )
#                     self.use_onnx = False
            
#             logger.info("Model loaded.")
#         return self._model

#     def encode(self, texts: Iterable[str]) -> np.ndarray:
#         texts_list: List[str] = list(texts)
#         logger.info(
#             "Encoding %d texts into embeddings (batch_size=%d, onnx=%s)",
#             len(texts_list),
#             self.batch_size,
#             self.use_onnx,
#         )
        
#         # Use optimized batch size for better throughput
#         embeddings = self.model.encode(
#             texts_list,
#             convert_to_numpy=True,
#             show_progress_bar=True,
#             batch_size=self.batch_size,
#             # ONNX is used automatically by sentence-transformers if available
#         )
        
#         return embeddings



# # pa-wJumiQf9ujNtF1O2qB1GrwRio7tenEaGsKkn8HobnYs


"""
Embedding utilities wrapping VoyageAI with a compatible interface.
"""

from typing import Iterable, List
import numpy as np
import voyageai

from .logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Thin wrapper around VoyageAI embeddings.

    Supports:
    - voyageai-3.5 embedding model
    - Batch processing
    - Drop-in replacement for sentence-transformers
    """

    def __init__(
        self,
        model_name: str = "voyageai-3.5",
        batch_size: int = 32,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize VoyageAI embedding model.

        Args:
            model_name: VoyageAI embedding model name
            batch_size: Number of texts per API call
            api_key: Optional API key (falls back to env var)
        """
        self.model_name = model_name
        self.batch_size = batch_size

        if api_key:
            voyageai.api_key = api_key

        self.client = voyageai.Client()

        logger.info("Initialized VoyageAI embedding model: %s", self.model_name)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list: List[str] = list(texts)

        logger.info(
            "Encoding %d texts using VoyageAI (model=%s, batch_size=%d)",
            len(texts_list),
            self.model_name,
            self.batch_size,
        )

        all_embeddings: List[List[float]] = []

        # Batch API calls for efficiency & cost control
        for i in range(0, len(texts_list), self.batch_size):
            batch = texts_list[i : i + self.batch_size]

            response = self.client.embed(
                texts=batch,
                model=self.model_name,
            )

            all_embeddings.extend(response.embeddings)

        return np.array(all_embeddings, dtype=np.float32)
