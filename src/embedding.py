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
Embedding utilities with support for multiple providers:
- sentence_transformers (default)
- voyage (VoyageAI API)
"""

from typing import Iterable, List, Tuple
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
    Useful to debug token limits & batching.
    """
    if not texts:
        return (0, 0, 0.0)
    lens = [len(t or "") for t in texts]
    return (min(lens), max(lens), sum(lens) / len(lens))


class EmbeddingModel:
    """
    Multi-provider embedding model wrapper.

    Supports:
    - sentence_transformers (HuggingFace models with optional ONNX)
    - voyage (VoyageAI API)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        provider: str = "sentence_transformers",
        batch_size: int = 32,
        device: str | None = None,
        use_onnx: bool = False,
        # Voyage-specific parameters
        voyage_api_key: str | None = None,
        voyage_input_type: str | None = None,
        voyage_truncation: bool | None = None,
        voyage_output_dimension: int | None = None,
        voyage_output_dtype: str = "float",
        voyage_timeout: int | None = None,
        voyage_max_retries: int = 2,
        # Model2vec-specific parameters
        model2vec_quantize: bool = False,
        model2vec_enable_wasm: bool = False,
    ) -> None:
        """
        Initialize embedding model.

        Args:
            model_name: Model name (HuggingFace or VoyageAI)
            provider: 'sentence_transformers', 'voyage', or 'model2vec'
            batch_size: Batch size for encoding
            device: Device for sentence_transformers ('cpu', 'cuda', None)
            use_onnx: Use ONNX Runtime for sentence_transformers
            voyage_api_key: VoyageAI API key
            voyage_input_type: VoyageAI input type ('query' or 'document')
            voyage_truncation: VoyageAI truncation setting
            voyage_output_dimension: VoyageAI output dimension
            voyage_output_dtype: VoyageAI output dtype
            voyage_timeout: VoyageAI timeout in seconds
            voyage_max_retries: VoyageAI max retries
            model2vec_quantize: Enable quantization for model2vec
            model2vec_enable_wasm: Enable WASM backend for model2vec
        """
        self.model_name = model_name
        self.provider = provider
        self.batch_size = batch_size
        self.device = device
        self.use_onnx = use_onnx
        self._model = None

        if provider == "sentence_transformers":
            self._init_sentence_transformers()
        elif provider == "voyage":
            self._init_voyage(
                api_key=voyage_api_key,
                input_type=voyage_input_type,
                truncation=voyage_truncation,
                output_dimension=voyage_output_dimension,
                output_dtype=voyage_output_dtype,
                timeout=voyage_timeout,
                max_retries=voyage_max_retries,
            )
        elif provider == "model2vec":
            self._init_model2vec(
                quantize=model2vec_quantize,
                enable_wasm=model2vec_enable_wasm,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _init_sentence_transformers(self) -> None:
        """Initialize sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = None
            logger.info(
                "Initialized sentence_transformers (model=%s, device=%s, onnx=%s)",
                self.model_name,
                self.device,
                self.use_onnx,
            )
        except ImportError:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

    def _init_voyage(
        self,
        api_key: str | None = None,
        input_type: str | None = None,
        truncation: bool | None = None,
        output_dimension: int | None = None,
        output_dtype: str = "float",
        timeout: int | None = None,
        max_retries: int = 2,
    ) -> None:
        """Initialize VoyageAI client."""
        try:
            import voyageai
        except ImportError:
            raise ImportError("voyageai not installed. Install with: pip install voyageai")

        if api_key:
            voyageai.api_key = api_key

        self.voyage_client = voyageai.Client()
        self.voyage_input_type = input_type
        self.voyage_truncation = truncation
        self.voyage_output_dimension = output_dimension
        self.voyage_output_dtype = output_dtype
        self.voyage_timeout = timeout
        self.voyage_max_retries = max_retries

        logger.info(
            "Initialized VoyageAI (model=%s, input_type=%s, output_dim=%s, dtype=%s)",
            self.model_name,
            input_type,
            output_dimension,
            output_dtype,
        )

    def _init_model2vec(
        self,
        quantize: bool = False,
        enable_wasm: bool = False,
    ) -> None:
        """Initialize model2vec embedding model."""
        try:
            from model2vec import StaticModel
        except ImportError:
            raise ImportError("model2vec not installed. Install with: pip install model2vec")

        self.model2vec_quantize = quantize
        self.model2vec_enable_wasm = enable_wasm
        self._m2v_model = None

        logger.info(
            "Initialized model2vec (model=%s, quantize=%s, enable_wasm=%s)",
            self.model_name,
            quantize,
            enable_wasm,
        )

    @property
    def model(self):
        """Lazy-load sentence-transformers model."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence_transformers model '%s'...", self.model_name)
            model_kwargs = {}
            if self.device:
                model_kwargs["device"] = self.device
            self._st_model = SentenceTransformer(self.model_name, **model_kwargs)
            logger.info("Model loaded.")
        return self._st_model

    @property
    def model2vec_model(self):
        """Lazy-load model2vec model."""
        if self._m2v_model is None:
            from model2vec import StaticModel
            import inspect
            print(inspect.signature(StaticModel.from_pretrained))
            logger.info("Loading model2vec model '%s'...", self.model_name)
            self._m2v_model = StaticModel.from_pretrained(
                self.model_name,
                # quantize=self.model2vec_quantize,
                # enable_wasm=self.model2vec_enable_wasm,
            )
            logger.info("Model loaded.")
        return self._m2v_model

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        texts_list: List[str] = list(texts)

        if self.provider == "sentence_transformers":
            return self._encode_sentence_transformers(texts_list)
        elif self.provider == "voyage":
            return self._encode_voyage(texts_list)
        elif self.provider == "model2vec":
            return self._encode_model2vec(texts_list)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _encode_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """Encode using sentence-transformers."""
        logger.info(
            "Encoding %d texts with sentence_transformers (batch_size=%d, onnx=%s)",
            len(texts),
            self.batch_size,
            self.use_onnx,
        )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=self.batch_size,
        )

        logger.info("Embeddings shape: %s", _safe_shape(embeddings))
        return embeddings

    def _encode_voyage(self, texts: List[str]) -> np.ndarray:
        """Encode using VoyageAI."""
        logger.info(
            "Encoding %d texts with VoyageAI (model=%s, batch_size=%d)",
            len(texts),
            self.model_name,
            self.batch_size,
        )

        all_embeddings: List[List[float]] = []

        # Batch API calls
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            embed_kwargs = {
                "texts": batch,
                "model": self.model_name,
            }
            if self.voyage_input_type:
                embed_kwargs["input_type"] = self.voyage_input_type
            if self.voyage_truncation is not None:
                embed_kwargs["truncation"] = self.voyage_truncation
            if self.voyage_output_dimension:
                embed_kwargs["output_dimension"] = self.voyage_output_dimension
            if self.voyage_output_dtype and self.voyage_output_dtype != "float":
                embed_kwargs["output_dtype"] = self.voyage_output_dtype

            response = self.voyage_client.embed(**embed_kwargs)
            all_embeddings.extend(response.embeddings)

        result = np.array(all_embeddings, dtype=np.float32)
        logger.info("Embeddings shape: %s", _safe_shape(result))
        return result
    def _encode_model2vec(self, texts: List[str]) -> np.ndarray:
        """Encode using model2vec."""
        logger.info(
            "Encoding %d texts with model2vec (model=%s, batch_size=%d)",
            len(texts),
            self.model_name,
            self.batch_size,
        )

        all_embeddings = []

        # Process texts in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self.model2vec_model.encode(batch)
            
            # Ensure it's a list of embeddings
            if isinstance(batch_embeddings, np.ndarray):
                if batch_embeddings.ndim == 1:
                    batch_embeddings = [batch_embeddings]
                all_embeddings.extend(batch_embeddings)
            else:
                all_embeddings.extend(batch_embeddings)

        result = np.array(all_embeddings, dtype=np.float32)
        logger.info("Embeddings shape: %s", _safe_shape(result))
        return result