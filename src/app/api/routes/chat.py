from fastapi import APIRouter, Request

from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
)


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    return ChatResponse(
        message=f"Mensaje recibido: {payload.message}"
    )
# def chat(
#     payload: ChatRequest,
#     request: Request,
# ) -> ChatResponse:
#     rag_service = request.app.state.rag_service

#     response = rag_service.ask(
#         payload.message,
#     )

#     return ChatResponse(
#         message=response,
#     )