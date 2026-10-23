import base64
import io
import secrets
from pathlib import Path
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from groq import Groq

from app.config import settings
from app.agent import graph


ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Medical Agent Educational API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.allowed_origins.split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Conversation-ID"],
)

# app.mount(
#     "/static",
#     StaticFiles(directory=ROOT / "static"),
#     name="static",
# )


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(
        default_factory=list,
        max_length=10,
    )
    image_data: str | None = None


def get_groq_client():
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="Set GROQ_API_KEY in your .env file.",
        )

    return Groq(
        api_key=settings.groq_api_key,
        timeout=45.0,
    )



def auth(authorization: str | None = Header(default=None)):
    configured_token = settings.api_token.strip()

    # Allow requests without a token only when API_TOKEN is unset.
    if not configured_token:
        return

    expected = f"Bearer {configured_token}"

    if not secrets.compare_digest(
        (authorization or "").strip(),
        expected,
    ):
        print("Auth failed: header present =", bool(authorization))
        print("Auth failed: configured token present =", bool(configured_token))

        raise HTTPException(
            status_code=401,
            detail="Invalid API token",
        )
    


def inspect_image(data: str | None) -> str:
    print("inspect_image called")

    if not data:
        return ""

    try:
        header, payload = data.split(",", 1)
        mime = header.split(";")[0].split(":")[1]

        if mime not in (
            "image/jpeg",
            "image/png",
            "image/webp",
        ):
            raise ValueError("Only JPEG, PNG, and WebP are supported")

        raw = base64.b64decode(payload, validate=True)

        if len(raw) > 5_000_000:
            raise ValueError("Image must be under 5 MB")

    except (ValueError, IndexError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        )

    try:
        client = get_groq_client()

        result = client.chat.completions.create(
            model=settings.groq_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe only objectively visible "
                                "medical-relevant features. Do not "
                                "diagnose, estimate severity, or claim "
                                "that an illness is absent. If the image "
                                "is unclear, say so."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data},
                        },
                    ],
                }
            ],
            max_completion_tokens=500,
        )

        return result.choices[0].message.content or ""

    except HTTPException:
        raise
    except Exception as exc:
        print("Groq vision error:", repr(exc), flush=True)
        raise HTTPException(
            status_code=502,
            detail="Image analysis failed. Check the Groq vision model and API key.",
        )


@app.get("/", response_class=HTMLResponse)
def home():
    # index_file = ROOT / "templates" / "index.html"

    # if not index_file.exists():
    #     return HTMLResponse(
    #         "<h2>MedGuide AI API is running.</h2>"
    #         "<p>Open your Next.js frontend on port 3000.</p>"
    #     )

    # return index_file.read_text(encoding="utf-8")
    return {
        "message": "MedGuide AI API is running",
        "frontend": "http://localhost:3000",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "configured": bool(
            settings.groq_api_key
            and settings.pinecone_api_key
        ),
        "tavily": bool(settings.tavily_api_key),
        "provider": "groq",
    }


@app.post("/api/chat")
def chat(
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Header(
        default=None,
        alias="X-Conversation-ID",
    ),
):
    auth(authorization)

    if not settings.groq_api_key or not settings.pinecone_api_key:
        raise HTTPException(
            status_code=503,
            detail="Set GROQ_API_KEY and PINECONE_API_KEY in .env",
        )

    history = [
        item
        for item in payload.history
        if item.get("role") in ("user", "assistant")
        and isinstance(item.get("content"), str)
    ][:10]
    

    thread_id = conversation_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        observation = inspect_image(payload.image_data)

        state = graph.invoke(
            {
                "question": payload.message,
                "history": history,
                "image_description": observation,
                "kb": [],
                "web": [],
                "answer": "",
                "route": "",
            },
            config=config,
        )

        return {
            "answer": state["answer"],
            "sources": [
                {
                    "type": "pdf",
                    "page": item["page"],
                    "score": item["score"],
                }
                for item in state["kb"]
            ] + [
                {
                    "type": "web",
                    "url": item["url"],
                    "title": item["title"],
                }
                for item in state["web"]
            ],
            "tools": (
                ["pinecone"]
                + (
                    ["tavily"]
                    if state["route"] == "web"
                    and settings.tavily_api_key
                    else []
                )
                + (["vision"] if observation else [])
            ),
            "conversation_id": thread_id,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print("Chat error:", repr(exc), flush=True)
        raise HTTPException(
            status_code=502,
            detail=(
                "AI service failed. Check server logs, Groq, "
                "Pinecone, and graph configuration."
            ),
        )



@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    auth(authorization)

    data = await file.read(10_000_001)

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty.",
        )

    if len(data) > 10_000_000:
        raise HTTPException(
            status_code=413,
            detail="Audio exceeds 10 MB.",
        )

    filename = file.filename or "voice.webm"
    content_type = file.content_type or "audio/webm"

    try:
        client = get_groq_client()

        result = client.audio.transcriptions.create(
            model=settings.groq_transcription_model,
            file=(filename, data, content_type),
            response_format="json",
        )

        transcript = (result.text or "").strip()

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="No speech detected. Please try again.",
            )

        return {"text": transcript}

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "Groq transcription error:",
            type(exc).__name__,
            str(exc),
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail="Speech transcription failed. Check backend logs.",
        )


@app.post("/api/speak")
def speak(
    payload: dict,
    authorization: str | None = Header(default=None),
):
    auth(authorization)

    text = str(payload.get("text", "")).strip()[:200]

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Missing text",
        )

    try:
        client = get_groq_client()

        speech = client.audio.speech.create(
            model=settings.groq_tts_model,
            voice="austin",
            input=text,
            response_format="wav",
        )

        return Response(
            content=speech.content,
            media_type="audio/wav",
        )

    except Exception as exc:
        print("Groq speech error:", repr(exc), flush=True)
        raise HTTPException(
            status_code=502,
            detail="Text-to-speech failed.",
        )




