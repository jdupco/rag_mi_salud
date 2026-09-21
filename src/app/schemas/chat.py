from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="Mensaje enviado por el usuario",
    )
    conversation_id: str = Field(
        ...,
        description="Identificador de la conversación",
    )


class ChatResponse(BaseModel):
    message: str = Field(
        ...,
        description="Respuesta generada por el asistente",
    )