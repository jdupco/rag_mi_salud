import os
from loguru import logger
from llama_index.core import StorageContext, load_index_from_storage
from config.settings import Config, init_llama_settings


class PDFQueryEngine:
    """Servicio para recargar el índice y procesar consultas RAG."""

    def __init__(self, storage_dir: str = Config.STORAGE_DIR):
        self.storage_dir = storage_dir
        self._query_engine = None
        init_llama_settings()  # Aplica los mismos embeddings antes de recuperar el vector

    def load_storage(self):
        if not os.path.exists(self.storage_dir) or not os.listdir(self.storage_dir):
            logger.error("No hay un índice válido en '{}'", self.storage_dir)
            raise FileNotFoundError(
                f"El directorio '{self.storage_dir}' está vacío. Ejecuta la indexación primero."
            )
        logger.info("Cargando vectores persistidos desde '{}'...", self.storage_dir)
        storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
        index = load_index_from_storage(storage_context)
        self._query_engine = index.as_query_engine()
        logger.success("Motor de búsqueda cargado y listo.")

    def ask(self, question: str) -> str:
        if self._query_engine is None:
            self.load_storage()

        logger.debug("Consultando RAG: '{}'", question)
        respuesta = self._query_engine.query(question)
        return str(respuesta)