"""FastAPI service with streaming support."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Multilingual Support Bot")

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 256

@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    response = generate_response(request.message)
    return {"response": response}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint with SSE."""
    async def generate():
        for token in stream_tokens(request.message):
            yield f"data: {token}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
