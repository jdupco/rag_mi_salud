from app.services.query_engine import PDFQueryEngine


class RAGService:

    def __init__(self):
        self.engine = PDFQueryEngine()
        self.engine.load_storage()

    def ask(self, message: str) -> str:
        return self.engine.ask(message)