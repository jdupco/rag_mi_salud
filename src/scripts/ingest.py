import os
from pathlib import Path

import pymupdf4llm
from loguru import logger

from llama_index.core import (
    Document,
    VectorStoreIndex,
)
from llama_index.core.node_parser import MarkdownNodeParser

from config.settings import init_llama_settings, Config


def run_ingestion():
    # 1. Configuración de LlamaIndex
    init_llama_settings()
    config = Config()

    documents_dir = Path("data/documents")
    storage_dir = config.STORAGE_DIR

    if not documents_dir.exists() or not any(documents_dir.iterdir()):
        logger.error(
            f"No se encontraron documentos en {documents_dir}. "
            "Agrega tus PDFs ahí."
        )
        return

    # 2. Convertir PDFs → Markdown
    logger.info("Convirtiendo PDFs a Markdown con PyMuPDF4LLM...")

    documents = []

    for pdf_path in documents_dir.glob("*.pdf"):

        logger.info(f"Procesando: {pdf_path.name}")

        markdown = pymupdf4llm.to_markdown(str(pdf_path))

        if not markdown.strip():
            logger.warning(f"PDF vacío: {pdf_path.name}")
            continue

        # Metadata de negocio/documento
        document = Document(
            text=markdown,
            metadata={
                "document_id": pdf_path.stem,
                "source": pdf_path.name,
                "file_path": str(pdf_path),
                "document_type": "health_plan",
            },
        )

        documents.append(document)

    logger.info(
        f"Convertidos {len(documents)} PDFs a documentos Markdown."
    )

    if not documents:
        logger.error("No se pudieron procesar documentos.")
        return

    # 3. Markdown → Nodes
    logger.info("Generando nodes con MarkdownNodeParser...")

    parser = MarkdownNodeParser()

    nodes = parser.get_nodes_from_documents(documents)

    logger.info(f"Generados {len(nodes)} nodes.")

    # 4. Crear índice + embeddings
    logger.info(
        "Generando embeddings con "
        "nvidia/nemotron-3-embed-1b..."
    )

    index = VectorStoreIndex(
        nodes,
        show_progress=True,
    )

    # 5. Persistir
    logger.info(f"Guardando índice en {storage_dir}...")

    os.makedirs(storage_dir, exist_ok=True)

    index.storage_context.persist(
        persist_dir=storage_dir
    )

    logger.success("¡Ingestión completada con éxito!")


if __name__ == "__main__":
    run_ingestion()