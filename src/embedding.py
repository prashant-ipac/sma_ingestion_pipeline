"""
Embedding utilities wrapping:
- SentenceTransformers (local)
- Voyage embeddings (API)
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)


def _safe_shape(arr) -> str:
    try:
        return str(arr.shape)
    except Exception:
        return "<unknown>"


def _text_stats(texts: List[str]) -> Tuple[int, int, float]:
    """
    Returns (min_len, max_len, avg_len) in characters.
    Useful to debug Voyage token limit errors & batching.
    """
    if not texts:
        return (0, 0, 0.0)
    lens = [len(t or "") for t in texts]
    return (min(lens), max(lens), sum(lens) / len(lens))


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

        logger.debug(
            "EmbeddingModel init: provider=%s model=%s batch_size=%s device=%s onnx=%s voyage_input_type=%s truncation=%s out_dim=%s dtype=%s timeout=%s retries=%s",
            self.provider,
            self.model_name,
            self.batch_size,
            self.device,
            self.use_onnx,
            self.voyage_input_type,
            self.voyage_truncation,
            self.voyage_output_dimension,
            self.voyage_output_dtype,
            self.voyage_timeout,
            self.voyage_max_retries,
        )

    # ---------------------------
    # SentenceTransformers
    # ---------------------------
    def _get_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            t0 = time.time()
            logger.info("Loading SentenceTransformer model '%s'...", self.model_name)
            kwargs = {}
            if self.device:
                kwargs["device"] = self.device

            self._st_model = SentenceTransformer(self.model_name, **kwargs)

            if self.use_onnx:
                # sentence-transformers can leverage ONNX if you set it up; leaving as-is.
                logger.info("ONNX mode enabled (SentenceTransformer path).")

            logger.info("SentenceTransformer model loaded in %.2fs.", time.time() - t0)
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

            logger.info(
                "Initializing Voyage client (timeout=%s, max_retries=%s)...",
                self.voyage_timeout,
                self.voyage_max_retries,
            )

            self._voyage_client = voyageai.Client(
                api_key=self.voyage_api_key,
                timeout=self.voyage_timeout,
                max_retries=self.voyage_max_retries,
            )
        return self._voyage_client

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list: List[str] = list(texts)
        if not texts_list:
            logger.warning("encode() called with 0 texts. Returning empty embedding array.")
            return np.zeros((0, 0), dtype=np.float32)

        min_len, max_len, avg_len = _text_stats(texts_list)
        logger.info(
            "encode(): provider=%s model=%s texts=%d (char_len min=%d max=%d avg=%.1f)",
            self.provider,
            self.model_name,
            len(texts_list),
            min_len,
            max_len,
            avg_len,
        )

        t0 = time.time()
        if self.provider == "voyage":
            out = self._encode_voyage(texts_list)
        else:
            out = self._encode_sentence_transformers(texts_list)

        logger.info(
            "encode() done: provider=%s embeddings_shape=%s dtype=%s total_time=%.2fs",
            self.provider,
            _safe_shape(out),
            getattr(out, "dtype", None),
            time.time() - t0,
        )
        return out

    def _encode_sentence_transformers(self, texts_list: List[str]) -> np.ndarray:
        logger.info(
            "Encoding with SentenceTransformer (texts=%d batch_size=%d device=%s onnx=%s)",
            len(texts_list),
            self.batch_size,
            self.device,
            self.use_onnx,
        )
        model = self._get_st_model()

        t0 = time.time()
        embeddings = model.encode(
            texts_list,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=self.batch_size,
        )
        logger.info(
            "SentenceTransformer encoding complete: shape=%s time=%.2fs",
            _safe_shape(embeddings),
            time.time() - t0,
        )
        return embeddings

    def _dtype_to_numpy(self) -> np.dtype:
        # Voyage returns different formats depending on output_dtype.
        if self.voyage_output_dtype == "float":
            return np.float32
        if self.voyage_output_dtype == "int8":
            return np.int8
        if self.voyage_output_dtype in ("uint8", "ubinary"):
            return np.uint8
        if self.voyage_output_dtype == "binary":
            return np.int8
        return np.float32

    def _encode_voyage(self, texts_list: List[str]) -> np.ndarray:
        logger.info(
            "Encoding with Voyage (texts=%d model=%s batch_size=%d input_type=%s truncation=%s out_dim=%s dtype=%s)",
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
        total = len(texts_list)
        total_batches = (total + bs - 1) // bs

        all_embs = []
        for batch_idx in range(total_batches):
            start = batch_idx * bs
            end = min(start + bs, total)
            batch = texts_list[start:end]

            bmin, bmax, bavg = _text_stats(batch)

            logger.info(
                "Voyage batch %d/%d: items=%d (char_len min=%d max=%d avg=%.1f)",
                batch_idx + 1,
                total_batches,
                len(batch),
                bmin,
                bmax,
                bavg,
            )

            # Extra protective retry (helps for rate limits / transient errors)
            attempt = 0
            while True:
                attempt += 1
                t0 = time.time()
                try:
                    res = client.embed(
                        batch,
                        model=self.model_name,
                        input_type=self.voyage_input_type,
                        truncation=self.voyage_truncation,
                        output_dimension=self.voyage_output_dimension,
                        output_dtype=self.voyage_output_dtype,
                    )
                    dt = time.time() - t0
                    all_embs.extend(res.embeddings)
                    logger.info(
                        "Voyage batch %d/%d success: returned=%d time=%.2fs total_embs=%d",
                        batch_idx + 1,
                        total_batches,
                        len(res.embeddings),
                        dt,
                        len(all_embs),
                    )
                    break
                except Exception as e:
                    dt = time.time() - t0
                    logger.warning(
                        "Voyage batch %d/%d failed (attempt=%d time=%.2fs): %s",
                        batch_idx + 1,
                        total_batches,
                        attempt,
                        dt,
                        e,
                        exc_info=True,
                    )
                    # Stop if too many attempts
                    if attempt >= 3:
                        logger.error(
                            "Voyage failed after %d attempts on batch %d/%d (items=%d).",
                            attempt,
                            batch_idx + 1,
                            total_batches,
                            len(batch),
                        )
                        raise

                    # Simple backoff
                    sleep_s = 1.5 * attempt
                    logger.info("Retrying Voyage batch %d/%d in %.1fs...", batch_idx + 1, total_batches, sleep_s)
                    time.sleep(sleep_s)

        arr = np.asarray(all_embs, dtype=self._dtype_to_numpy())
        logger.info("Voyage encoding complete: shape=%s", _safe_shape(arr))
        return arr
