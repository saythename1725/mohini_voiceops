"""File-based Gnani speech-to-text and text-to-speech pipeline."""

import os
from pathlib import Path


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _transcribe_audio(source: Path, language_code: str) -> str:
    from gnani.stt import GnaniSTTClient

    result = GnaniSTTClient().transcribe(str(source), language_code=language_code)
    return str(result.get("transcript", "")).strip()


def _synthesize_audio(text: str, output_path: Path, language_code: str) -> None:
    from gnani.tts import AudioConfig, GnaniTTSClient

    audio = GnaniTTSClient().synthesize(
        text,
        voice=os.getenv("GNANI_TTS_VOICE", "Kaveri"),
        model="timbre-v2.5",
        language=language_code,
        audio_config=AudioConfig(sample_rate=24000, encoding="linear_pcm", container="wav"),
    )
    # Gnani may return a streaming WAV header with 0xFFFFFFFF size markers.
    # Those files can be saved, but Windows' WAV playback APIs treat them as invalid.
    audio = bytearray(audio)
    if len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        audio[4:8] = (len(audio) - 8).to_bytes(4, "little")
        offset = 12
        while offset + 8 <= len(audio):
            chunk_id = bytes(audio[offset:offset + 4])
            chunk_size = int.from_bytes(audio[offset + 4:offset + 8], "little")
            if chunk_id == b"data":
                audio[offset + 4:offset + 8] = (len(audio) - offset - 8).to_bytes(4, "little")
                break
            offset += 8 + chunk_size + (chunk_size & 1)
    output_path.write_bytes(audio)


def voice_turn(audio_path: str | Path, agent, output_dir: str | Path,
               language_code: str = "en-IN") -> dict:
    """Transcribe one recording, run the RAG agent, synthesize speech, and return artifacts."""
    if not os.getenv("GNANI_API_KEY"):
        raise RuntimeError("Set GNANI_API_KEY in voiceops_agent/.env before using voice mode.")

    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")
    if source.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        formats = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio type {source.suffix!r}; supported formats: {formats}.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transcript_path = output / f"{source.stem}_transcript.txt"
    reply_audio_path = output / f"{source.stem}_reply.wav"

    transcript = _transcribe_audio(source, language_code)
    transcript_path.write_text(transcript + "\n", encoding="utf-8")

    if transcript:
        answer = agent.respond(transcript)
    else:
        retry_reply = ("माफ़ कीजिए, आवाज़ साफ़ नहीं सुनाई दी। कृपया फिर से कहें।"
                       if language_code.startswith("hi") else
                       "Sorry, I couldn't hear that clearly. Please say it again.")
        answer = {
            "reply": retry_reply, "sources": [], "decision": "clarify", "tool_calls": [],
            "ticket_created": False, "ticket_created_this_turn": False,
            "order_verified": False, "llm_error": None, "llm_used": False,
        }

    audio_error = None
    try:
        _synthesize_audio(answer["reply"], reply_audio_path, language_code)
    except Exception as exc:
        audio_error = str(exc)
    return {
        "input_audio": str(source),
        "transcript": transcript,
        **answer,
        "transcript_file": str(transcript_path),
        "audio_reply": str(reply_audio_path) if audio_error is None else None,
        "audio_error": audio_error,
    }
