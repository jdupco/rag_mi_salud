import os
import sys
import warnings
from dotenv import load_dotenv

# Silenciar advertencias de Hugging Face en consola
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning)

from loguru import logger
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

load_dotenv()

# Variables de entorno críticas
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


class Config:
    """Configuración centralizada del sistema."""

    OPENCODE_API_KEY: str = os.getenv("OPENCODE_API_KEY", "")
    OPENCODE_BASE_URL: str = os.getenv(
        "OPENCODE_BASE_URL", "https://opencode.ai/zen/v1/"
    )
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-v4-flash")
    EMBED_MODEL_NAME: str = os.getenv(
        "EMBED_MODEL_NAME", "intfloat/multilingual-e5-small"
    )

    DATA_DIR: str = os.getenv("DATA_DIR", "./mis_pdfs")
    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "./storage")
    LOG_DIR: str = os.getenv("LOG_DIR", "./logs")


def setup_logger():
    """Configura Loguru para consola y archivos rotatorios."""
    logger.remove()

    # Consola limpia (corregido {level:<8})
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Configuración de archivos
    if not os.path.exists(Config.LOG_DIR):
        os.makedirs(Config.LOG_DIR)

    logger.add(
        os.path.join(Config.LOG_DIR, "rag_app.log"),
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        level="DEBUG",
        enqueue=True,
    )


def init_llama_settings():
    """Inicializa la configuración global de LlamaIndex."""
    setup_logger()

    if not Config.OPENCODE_API_KEY:
        logger.error(
            "La variable OPENCODE_API_KEY no está definida en el archivo .env"
        )
        raise ValueError("OPENCODE_API_KEY faltante en .env")

    logger.info("Cargando modelo de Embeddings: {}", Config.EMBED_MODEL_NAME)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBED_MODEL_NAME
    )

    logger.info("Cargando LLM: {}", Config.LLM_MODEL)
    Settings.llm = OpenAILike(
        model=Config.LLM_MODEL,
        api_key=Config.OPENCODE_API_KEY,
        api_base=Config.OPENCODE_BASE_URL,
        is_chat_model=True,
        context_window=128000,
        temperature=0.1,
    )
    logger.success("Entorno de LlamaIndex configurado correctamente.")