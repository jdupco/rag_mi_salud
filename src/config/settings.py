import os
import sys
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
from loguru import logger

load_dotenv()


class Config:
    """Configuración centralizada del sistema."""

    # LLM (OpenCode)
    OPENCODE_API_KEY: str = os.getenv("OPENCODE_API_KEY", "")
    OPENCODE_BASE_URL: str = os.getenv(
        "OPENCODE_BASE_URL", "https://opencode.ai/zen/v1/"
    )
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-v4-flash")

    # Embeddings (NVIDIA NIM API via HTTP)
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_EMBED_BASE_URL: str = os.getenv(
        "NVIDIA_EMBED_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
    EMBED_MODEL_NAME: str = os.getenv(
        "EMBED_MODEL_NAME", "nvidia/nemotron-3-embed-1b"
    )

    # Rutas
    DATA_DIR: str = os.getenv("DATA_DIR", "data/documents")
    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "data/storage")
    LOG_DIR: str = os.getenv("LOG_DIR", "./logs")


def setup_logger():
    """Configura Loguru para consola y archivos rotatorios."""
    logger.remove()

    # Consola
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Archivo de logs
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
    """Inicializa la configuración global de LlamaIndex mediante llamadas HTTP remotas."""
    setup_logger()

    # Validaciones de variables de entorno requeridas
    if not Config.OPENCODE_API_KEY:
        logger.error("OPENCODE_API_KEY no está definida en las variables de entorno.")
        raise ValueError("OPENCODE_API_KEY faltante en .env")

    if not Config.NVIDIA_API_KEY:
        logger.error("NVIDIA_API_KEY no está definida en las variables de entorno.")
        raise ValueError("NVIDIA_API_KEY faltante en .env")

    # Embedding remoto (NVIDIA NIM)
    logger.info("Configurando Embeddings remotos (NVIDIA): {}", Config.EMBED_MODEL_NAME)
    Settings.embed_model = OpenAILikeEmbedding(
        model_name=Config.EMBED_MODEL_NAME,
        api_key=Config.NVIDIA_API_KEY,
        api_base=Config.NVIDIA_EMBED_BASE_URL,
    )

    # LLM remoto (OpenCode)
    logger.info("Configurando LLM remoto: {}", Config.LLM_MODEL)
    Settings.llm = OpenAILike(
        model=Config.LLM_MODEL,
        api_key=Config.OPENCODE_API_KEY,
        api_base=Config.OPENCODE_BASE_URL,
        is_chat_model=True,
        context_window=128000,
        temperature=0.1,
    )

    logger.success("Entorno de LlamaIndex configurado correctamente sin dependencias locales pesadas.")