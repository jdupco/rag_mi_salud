import os
from loguru import logger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from config.settings import Config, init_llama_settings


class PDFIndexer:
    """Servicio para ingesta y vectorización de PDFs."""

    def __init__(self, data_dir: str = Config.DATA_DIR, storage_dir: str = Config.STORAGE_DIR):
        self.data_dir = data_dir
        self.storage_dir = storage_dir
        init_llama_settings()  # Garantiza que Settings.embed_model y Settings.llm estén listos

    def index_documents(self) -> VectorStoreIndex:
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.error("No existe el directorio de origen: {}", self.data_dir)
            raise FileNotFoundError(f"Carpeta '{self.data_dir}' creada. Coloca tus PDFs ahí.")

        logger.info("Leyendo PDFs desde '{}'...", self.data_dir)
        documents = SimpleDirectoryReader(
            input_dir=self.data_dir, required_exts=[".pdf"]
        ).load_data()

        if not documents:
            logger.warning("No se encontraron PDFs en '{}'", self.data_dir)
            raise ValueError(f"Agrega al menos un PDF en '{self.data_dir}'.")

        logger.info("Generando vectores para {} documentos...", len(documents))
        index = VectorStoreIndex.from_documents(documents)

        logger.info("Guardando índice en disco ('{}')...", self.storage_dir)
        index.storage_context.persist(persist_dir=self.storage_dir)
        logger.success("Indexación completada y guardada con éxito.")
        return index