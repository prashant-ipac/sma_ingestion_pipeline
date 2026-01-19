# """
# Embedding utilities wrapping Sentence Transformers OR VoyageAI embeddings.
# """

# from __future__ import annotations

# import os
# import time
# from typing import Iterable, List, Optional

# import numpy as np
# from sentence_transformers import SentenceTransformer

# from .logging_utils import get_logger

# logger = get_logger(__name__)


# class EmbeddingModel:
#     """
#     Thin wrapper around:
#     - SentenceTransformer (local inference)
#     - VoyageAI embeddings (API)

#     Auto-selects Voyage when model_name starts with "voyage-".
#     """

#     def __init__(
#         self,
#         model_name: str,
#         use_onnx: bool = False,
#         batch_size: int = 32,
#         device: str | None = None,
#         # Voyage-specific overrides (optional)
#         voyage_api_key: str | None = None,
#         voyage_input_type: str | None = None,   # "query" | "document" | None
#         voyage_truncation: bool | None = None,
#         voyage_output_dimension: int | None = None,
#         voyage_output_dtype: str = "float",     # "float" | "int8" | "uint8" | "binary" | "ubinary"
#         voyage_max_retries: int | None = None,
#         voyage_timeout: int | None = None,
#     ) -> None:
#         self.model_name = model_name
#         self.use_onnx = use_onnx
#         self.batch_size = batch_size
#         self.device = device

#         self._st_model: SentenceTransformer | None = None

#         # Voyage config (read env if not explicitly passed)
#         self._use_voyage = model_name.startswith("voyage-")
#         self._voyage_client = None

#         self.voyage_api_key = voyage_api_key or os.getenv("VOYAGE_API_KEY")
#         self.voyage_input_type = voyage_input_type or os.getenv("VOYAGE_INPUT_TYPE") or None

#         vt = os.getenv("VOYAGE_TRUNCATION")
#         self.voyage_truncation = voyage_truncation if voyage_truncation is not None else (
#             (vt.lower() == "true") if vt else None
#         )

#         vod = os.getenv("VOYAGE_OUTPUT_DIMENSION")
#         self.voyage_output_dimension = voyage_output_dimension if voyage_output_dimension is not None else (
#             int(vod) if vod else None
#         )

#         self.voyage_output_dtype = os.getenv("VOYAGE_OUTPUT_DTYPE", voyage_output_dtype)

#         vmr = os.getenv("VOYAGE_MAX_RETRIES")
#         self.voyage_max_retries = voyage_max_retries if voyage_max_retries is not None else (
#             int(vmr) if vmr else 0
#         )

#         vto = os.getenv("VOYAGE_TIMEOUT")
#         self.voyage_timeout = voyage_timeout if voyage_timeout is not None else (
#             int(vto) if vto else None
#         )

#         if self._use_voyage and not self.voyage_api_key:
#             raise ValueError("VOYAGE_API_KEY is required when using a Voyage model (model_name starts with 'voyage-').")

#     @property
#     def st_model(self) -> SentenceTransformer:
#         if self._st_model is None:
#             logger.info("Loading SentenceTransformer model '%s'...", self.model_name)

#             model_kwargs = {}
#             if self.device:
#                 model_kwargs["device"] = self.device

#             self._st_model = SentenceTransformer(self.model_name, **model_kwargs)

#             if self.use_onnx:
#                 logger.info("ONNX mode enabled (SentenceTransformer path).")

#             logger.info("SentenceTransformer model loaded.")
#         return self._st_model

#     @property
#     def voyage_client(self):
#         if self._voyage_client is None:
#             try:
#                 import voyageai
#             except ImportError as e:
#                 raise ImportError("Missing dependency: voyageai. Install with: pip install -U voyageai") from e

#             # Client reads VOYAGE_API_KEY automatically if api_key=None,
#             # but we pass it explicitly for clarity.
#             # Client supports max_retries + timeout. :contentReference[oaicite:5]{index=5}
#             self._voyage_client = voyageai.Client(
#                 api_key=self.voyage_api_key,
#                 max_retries=self.voyage_max_retries or 0,
#                 timeout=self.voyage_timeout,
#             )
#         return self._voyage_client

#     def encode(self, texts: Iterable[str]) -> np.ndarray:
#         texts_list: List[str] = list(texts)
#         if not texts_list:
#             return np.zeros((0, 0), dtype=np.float32)

#         if self._use_voyage:
#             return self._encode_voyage(texts_list)

#         return self._encode_sentence_transformers(texts_list)

#     def _encode_sentence_transformers(self, texts_list: List[str]) -> np.ndarray:
#         logger.info(
#             "Encoding %d texts with SentenceTransformer (batch_size=%d, onnx=%s)",
#             len(texts_list),
#             self.batch_size,
#             self.use_onnx,
#         )
#         return self.st_model.encode(
#             texts_list,
#             convert_to_numpy=True,
#             show_progress_bar=True,
#             batch_size=self.batch_size,
#         )

#     def _encode_voyage(self, texts_list: List[str]) -> np.ndarray:
#         # Voyage API constraints: list length <= 1000. :contentReference[oaicite:6]{index=6}
#         bs = min(max(1, int(self.batch_size)), 1000)

#         logger.info(
#             "Encoding %d texts with Voyage (model=%s, batch_size=%d, input_type=%s, truncation=%s, out_dim=%s, dtype=%s)",
#             len(texts_list),
#             self.model_name,
#             bs,
#             self.voyage_input_type,
#             self.voyage_truncation,
#             self.voyage_output_dimension,
#             self.voyage_output_dtype,
#         )

#         all_embs = []
#         for i in range(0, len(texts_list), bs):
#             batch = texts_list[i : i + bs]

#             # Simple extra retry loop (in addition to Client(max_retries=...))
#             attempt = 0
#             while True:
#                 try:
#                     # Client.embed(texts, model, input_type, truncation, output_dimension, output_dtype) :contentReference[oaicite:7]{index=7}
#                     res = self.voyage_client.embed(
#                         batch,
#                         model=self.model_name,
#                         input_type=self.voyage_input_type,
#                         truncation=self.voyage_truncation,
#                         output_dimension=self.voyage_output_dimension,
#                         output_dtype=self.voyage_output_dtype,
#                     )
#                     all_embs.extend(res.embeddings)
#                     break
#                 except Exception as e:
#                     attempt += 1
#                     if attempt > 2:
#                         raise
#                     wait_s = 1.5 * attempt
#                     logger.warning("Voyage embed failed (attempt %d). Retrying in %.1fs. Error: %s", attempt, wait_s, e)
#                     time.sleep(wait_s)

#         # Map dtype
#         if self.voyage_output_dtype == "float":
#             return np.asarray(all_embs, dtype=np.float32)
#         if self.voyage_output_dtype == "int8":
#             return np.asarray(all_embs, dtype=np.int8)
#         if self.voyage_output_dtype in ("uint8", "ubinary"):
#             return np.asarray(all_embs, dtype=np.uint8)

#         # "binary" is bit-packed int8 lists (docs). :contentReference[oaicite:8]{index=8}
#         if self.voyage_output_dtype == "binary":
#             return np.asarray(all_embs, dtype=np.int8)

#         # Fallback
#         return np.asarray(all_embs, dtype=np.float32)


"""
Embedding utilities wrapping:
- SentenceTransformers (local)
- Voyage embeddings (API)
"""

from __future__ import annotations

from typing import Iterable, List
import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Wrapper around embeddings backends with lazy loading.

    Providers:
    - sentence_transformers (default)
    - voyage (VoyageAI API)
    """

    def __init__(
        self,
        model_name: str,
        provider: str = "sentence_transformers",
        batch_size: int = 32,
        device: str | None = None,
        use_onnx: bool = False,
        # Voyage params
        voyage_api_key: str | None = None,
        voyage_input_type: str | None = None,     # query|document|None
        voyage_truncation: bool | None = None,
        voyage_output_dimension: int | None = None,
        voyage_output_dtype: str = "float",       # float|int8|uint8|binary|ubinary
        voyage_timeout: int | None = None,
        voyage_max_retries: int = 2,
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        self.batch_size = batch_size
        self.device = device
        self.use_onnx = use_onnx

        self.voyage_api_key = voyage_api_key
        self.voyage_input_type = voyage_input_type
        self.voyage_truncation = voyage_truncation
        self.voyage_output_dimension = voyage_output_dimension
        self.voyage_output_dtype = voyage_output_dtype
        self.voyage_timeout = voyage_timeout
        self.voyage_max_retries = voyage_max_retries

        self._st_model = None
        self._voyage_client = None

    # ---------------------------
    # SentenceTransformers
    # ---------------------------
    def _get_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model '%s'...", self.model_name)
            kwargs = {}
            if self.device:
                kwargs["device"] = self.device

            self._st_model = SentenceTransformer(self.model_name, **kwargs)

            if self.use_onnx:
                # sentence-transformers can leverage ONNX if you set it up; leaving as-is.
                logger.info("ONNX mode enabled (SentenceTransformer path).")

            logger.info("SentenceTransformer model loaded.")
        return self._st_model

    # ---------------------------
    # Voyage
    # ---------------------------
    def _get_voyage_client(self):
        if self._voyage_client is None:
            try:
                import voyageai
            except ImportError as e:
                raise ImportError("voyageai not installed. Run: pip install -U voyageai") from e

            if not self.voyage_api_key:
                raise ValueError("VOYAGE_API_KEY is required for Voyage embeddings.")

            self._voyage_client = voyageai.Client(
                api_key=self.voyage_api_key,
                timeout=self.voyage_timeout,
                max_retries=self.voyage_max_retries,
            )
        return self._voyage_client

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list: List[str] = list(texts)
        if not texts_list:
            return np.zeros((0, 0), dtype=np.float32)

        if self.provider == "voyage":
            return self._encode_voyage(texts_list)

        return self._encode_sentence_transformers(texts_list)

    def _encode_sentence_transformers(self, texts_list: List[str]) -> np.ndarray:
        logger.info(
            "Encoding %d texts with SentenceTransformer (batch_size=%d, onnx=%s, device=%s)",
            len(texts_list),
            self.batch_size,
            self.use_onnx,
            self.device,
        )
        model = self._get_st_model()
        return model.encode(
            texts_list,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=self.batch_size,
        )

    def _dtype_to_numpy(self) -> np.dtype:
        # Voyage returns different formats depending on output_dtype.
        # Most common: float -> list[float]
        if self.voyage_output_dtype == "float":
            return np.float32
        if self.voyage_output_dtype == "int8":
            return np.int8
        if self.voyage_output_dtype in ("uint8", "ubinary"):
            return np.uint8
        if self.voyage_output_dtype == "binary":
            # binary is returned as int8 lists (bit-packed); keep as int8 array
            return np.int8
        return np.float32

    def _encode_voyage(self, texts_list: List[str]) -> np.ndarray:
        logger.info(
            "Encoding %d texts with Voyage (model=%s, batch_size=%d, input_type=%s, truncation=%s, out_dim=%s, dtype=%s)",
            len(texts_list),
            self.model_name,
            self.batch_size,
            self.voyage_input_type,
            self.voyage_truncation,
            self.voyage_output_dimension,
            self.voyage_output_dtype,
        )

        client = self._get_voyage_client()

        # Voyage supports up to 1000 texts per request; keep it bounded.
        bs = min(max(1, int(self.batch_size)), 1000)

        all_embs = []
        for i in range(0, len(texts_list), bs):
            batch = texts_list[i : i + bs]

            res = client.embed(
                batch,
                model=self.model_name,
                input_type=self.voyage_input_type,
                truncation=self.voyage_truncation,
                output_dimension=self.voyage_output_dimension,
                output_dtype=self.voyage_output_dtype,
            )

            all_embs.extend(res.embeddings)

        return np.asarray(all_embs, dtype=self._dtype_to_numpy())
