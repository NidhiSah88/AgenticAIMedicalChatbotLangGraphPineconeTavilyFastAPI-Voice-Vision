"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Message = {
  role: "user" | "assistant";
  content: string;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export default function MedGuideChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [message, setMessage] = useState("");
  const [status, setStatus] = useState("Checking...");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);

  const imageInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [imageData, setImageData] = useState<string | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [recording, setRecording] = useState(false);
  const [voiceLoading, setVoiceLoading] = useState(false);
  const [actionError, setActionError] = useState("");

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((response) => {
        if (!response.ok) throw new Error("Backend unavailable");
        return response.json();
      })
      .then((data) =>
        setStatus(data.configured ? "● Connected" : "⚠ Configure API keys"),
      )
      .catch(() => setStatus("● Server offline"));
  }, []);

  async function sendMessage(event: React.FormEvent) {
    console.log("Sending message: called ", message);
    event.preventDefault();

    const text = message.trim();
    if (!text || busy) return;

    const previous = messages.slice(-8);
    setMessages((current) => [...current, { role: "user", content: text }]);
    setMessage("");
    setBusy(true);

    try {
      console.log("Chat token present:", Boolean(token.trim()));
      console.log("Chat token length:", token.trim().length);
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token.trim() ? { Authorization: `Bearer ${token.trim()}` } : {}),
        },
        body: JSON.stringify({
          message: text,
          history: previous,
          image_data: imageData,
        }),
      });

      const data = await response.json();
      console.log("Sendingmessage () RESPONSE API data) ", data);

      if (!response.ok) {
        throw new Error(data.detail || "Request failed");
      }

      setMessages((current) => [
        ...current,
        { role: "assistant", content: data.answer },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content:
            error instanceof Error
              ? `Error: ${error.message}`
              : "Request failed",
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  function handleImageSelect(file?: File) {
    setActionError("");

    if (!file) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setActionError("Choose a JPEG, PNG, or WebP image.");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setActionError("Image must be smaller than 5 MB.");
      return;
    }

    const reader = new FileReader();

    reader.onload = () => {
      setImageData(reader.result as string);
    };

    reader.onerror = () => {
      setActionError("Could not read this image.");
    };

    reader.readAsDataURL(file);
  }

  async function openCamera() {
    setActionError("");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      streamRef.current = stream;
      setCameraOpen(true);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch {
      setActionError(
        "Camera access failed. Allow camera permission in your browser.",
      );
    }
  }

  function closeCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOpen(false);
  }

  function capturePhoto() {
    const video = videoRef.current;

    if (!video || !video.videoWidth) {
      setActionError("Camera is not ready yet.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    if (!context) {
      setActionError("Could not capture the image.");
      return;
    }

    context.drawImage(video, 0, 0);
    setImageData(canvas.toDataURL("image/jpeg", 0.85));
    closeCamera();
  }

  async function startVoiceRecording() {
    setActionError("");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";

      const recorder = new MediaRecorder(stream, { mimeType });

      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };

      recorder.onerror = () => {
        setActionError("Audio recording failed.");
        stream.getTracks().forEach((track) => track.stop());
        setRecording(false);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());

        const audioBlob = new Blob(chunksRef.current, {
          type: recorder.mimeType,
        });

        if (!audioBlob.size) {
          setActionError("No audio was recorded.");
          return;
        }

        const formData = new FormData();
        formData.append("file", audioBlob, "voice.webm");

        try {
          setVoiceLoading(true);

          console.log("Audio API URL:", `${API_BASE}/api/transcribe`);
          console.log("Audio token present:", Boolean(token.trim()));
          console.log("Audio token length:", token.trim().length);

          const response = await fetch(`${API_BASE}/api/transcribe`, {
            method: "POST",
            headers: {
              ...(token.trim()
                ? { Authorization: `Bearer ${token.trim()}` }
                : {}),
            },
            body: formData,
          });

          const result = await response.json();

          if (!response.ok) {
            throw new Error(result.detail || "Transcription failed.");
          }

          if (result.text) {
            setMessage((previous) =>
              previous ? `${previous} ${result.text}` : result.text,
            );
          } else {
            setActionError("No speech was detected.");
          }
        } catch (error) {
          setActionError(
            error instanceof Error ? error.message : "Transcription failed.",
          );
        } finally {
          setVoiceLoading(false);
        }
      };

      recorder.start();
      setRecording(true);
    } catch {
      setActionError(
        "Microphone access failed. Allow microphone permission in your browser.",
      );
    }
  }

  function stopVoiceRecording() {
    const recorder = recorderRef.current;

    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }

    setRecording(false);
  }

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());

      const recorder = recorderRef.current;

      if (recorder && recorder.state !== "inactive") {
        recorder.onstop = null;
        recorder.stop();
      }
    };
  }, []);

  return (
    <main className="app">
      <header className="appHeader">
        <div>
          <h1>✚ MedGuide AI</h1>
          <p>Medical education • Knowledge base + web research</p>
        </div>
        <span className="status">{status}</span>
      </header>

      <div className="notice">
        Educational guidance only. Not a diagnosis. For severe symptoms, contact
        emergency services.
      </div>

      <section className="messages" aria-live="polite">
        <article className="bot">
          Hello! Ask a medical question, upload an image, or use the microphone.
        </article>
        MESSAGE
        {/*  
        {messages.map((item, index) => (
          <article className={item.role} key={index}>
            {item.content}
          </article>
        ))} */}
        <div className="markdown-content">
          {messages.map((item, index) => {
            const content =
              typeof item.content === "string"
                ? item.content
                : item.content == null
                  ? ""
                  : String(item.content);

            return (
              <article className={item.role} key={index}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {content}
                </ReactMarkdown>
              </article>
            );
          })}
        </div>
        {busy && <article className="bot">Thinking…</article>}
      </section>

      {/* Hidden image picker */}
      <input
        ref={imageInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        hidden
        onChange={(event) => {
          handleImageSelect(event.target.files?.[0]);
          event.target.value = "";
        }}
      />

      {/* Selected image preview */}
      {imageData && (
        <div className="preview">
          <img
            src={imageData}
            alt="Selected medical image"
            className="thumbnail"
          />
          <button
            type="button"
            className="action"
            onClick={() => setImageData(null)}
          >
            Remove image ✕
          </button>
        </div>
      )}

      {/* Camera preview */}
      {cameraOpen && (
        <div className="cameraPanel">
          <video ref={videoRef} autoPlay playsInline muted />
          <div className="cameraActions">
            <button type="button" className="action" onClick={capturePhoto}>
              Capture photo
            </button>

            <button type="button" className="action" onClick={closeCamera}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Permission and API errors */}
      {actionError && (
        <p className="error" role="alert">
          {actionError}
        </p>
      )}

      <form className="chatForm" onSubmit={sendMessage}>
        <textarea
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          placeholder="Describe your symptoms or ask a question..."
          rows={2}
          maxLength={4000}
          required
        />

        <div className="toolbar">
          <button
            type="button"
            className="action"
            onClick={() => imageInputRef.current?.click()}
          >
            📎 Image
          </button>

          <button
            type="button"
            className="action"
            onClick={() => {
              if (cameraOpen) {
                closeCamera();
              } else {
                void openCamera();
              }
            }}
          >
            {cameraOpen ? "✕ Close Camera" : "📷 Camera"}
          </button>

          <button
            type="button"
            className="action"
            disabled={voiceLoading}
            onClick={() => {
              if (recording) {
                stopVoiceRecording();
              } else {
                void startVoiceRecording();
              }
            }}
          >
            {voiceLoading
              ? "⏳ Transcribing..."
              : recording
                ? "⏹ Stop Recording"
                : "🎤 Voice"}
          </button>

          <button
            type="button"
            className="action"
            disabled={!messages.some((item) => item.role === "bot")}
            onClick={() => {
              const lastAnswer = [...messages]
                .reverse()
                .find((item) => item.role === "bot");

              if (!lastAnswer?.content) return;

              if ("speechSynthesis" in window) {
                window.speechSynthesis.cancel();

                const utterance = new SpeechSynthesisUtterance(
                  lastAnswer.content,
                );

                utterance.lang = "en-IN";
                window.speechSynthesis.speak(utterance);
              } else {
                setActionError(
                  "Speech synthesis is not supported in this browser.",
                );
              }
            }}
          >
            🔊 Read answer
          </button>

          <button className="send" type="submit" disabled={busy}>
            {busy ? "Thinking…" : "Send ➜"}
          </button>
        </div>
      </form>

      <details className="apiDetails">
        <summary>API access (optional)</summary>
        <input
          type="password"
          placeholder="API_TOKEN from backend .env"
          value={token}
          onChange={(event) => setToken(event.target.value)}
          autoComplete="off"
        />
      </details>
    </main>
  );
}
