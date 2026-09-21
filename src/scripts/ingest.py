import os
from loguru import logger
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from config.settings import init_llama_settings, Config


def run_ingestion():
    # 1. Cargar la configuración de LlamaIndex (NVIDIA Embeddings + LLM)
    init_llama_settings()
    config = Config()

    documents_dir = "data/documents"  # Ajusta según la ubicación de tus PDFs
    storage_dir = config.STORAGE_DIR

    if not os.path.exists(documents_dir) or not os.listdir(documents_dir):
        logger.error(f"No se encontraron documentos en {documents_dir}. Agrega tus PDFs ahí.")
        return

    logger.info(f"Cargando documentos desde {documents_dir}...")
    reader = SimpleDirectoryReader(documents_dir)
    documents = reader.load_data()
    logger.info(f"Cargados {len(documents)} documentos/páginas.")

    logger.info("Generando embeddings con nvidia/nemotron-3-embed-1b y creando el índice...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    logger.info(f"Guardando el nuevo índice en {storage_dir}...")
    os.makedirs(storage_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=storage_dir)

    logger.success("¡Ingestión completada con éxito!")


if __name__ == "__main__":
    run_ingestion()