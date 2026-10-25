"""Run ordered audio clips as turns in one persistent support conversation."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

from src.agent import SupportAgent
from src.voice import SUPPORTED_AUDIO_EXTENSIONS, voice_turn


def play_audio(path: str | Path, label: str) -> None:
    """Play an input or output clip synchronously for a screen recording."""
    if sys.platform != "win32":
        print(f"Audio playback is Windows-only in this demo runner; saved: {path}", flush=True)
        return
    print(f"Playing {label}...", flush=True)
    if Path(path).suffix.lower() == ".wav":
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return
    player = shutil.which("ffplay")
    if not player:
        print("Cannot play this audio format: install FFmpeg and add ffplay to PATH. "
              f"The file is still available at {path}", flush=True)
        return
    subprocess.run(
        [player, "-nodisp", "-autoexit", "-loglevel", "error", str(path)],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run audio clips in filename order through one persistent agent."
    )
    parser.add_argument("input_folder", nargs="?", default="audio_inputs",
                        help="Folder of sequential audio clips (default: audio_inputs)")
    parser.add_argument("--language", default="en-IN",
                        help="Gnani locale, e.g. en-IN, hi-IN, or gu-IN")
    parser.add_argument("--no-playback", action="store_true",
                        help="Save audio outputs without playing them during the run")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.is_absolute():
        input_folder = APP_DIR / input_folder
    if not input_folder.is_dir():
        parser.error(f"input folder does not exist: {input_folder}")

    clips = sorted(
        (p for p in input_folder.iterdir()
         if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS),
        key=lambda p: p.name.casefold(),
    )
    if not clips:
        parser.error(f"no supported audio clips found in {input_folder}")

    output_dir = APP_DIR / "outputs" / "voice"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep this single instance for the entire demo so follow-up turns share state.
    agent = SupportAgent(use_llm=True, verbose=True)
    turns = []
    for index, clip in enumerate(clips, start=1):
        print(f"\n--- Turn {index}/{len(clips)}: {clip.name} ---", flush=True)
        if not args.no_playback:
            play_audio(clip, "customer question")
        result = voice_turn(clip, agent, output_dir, language_code=args.language)
        turns.append(result)
        print(f"Customer: {result['transcript']}", flush=True)
        print(f"Assistant: {result['reply']}", flush=True)
        print(f"Decision: {result.get('decision')}", flush=True)
        print(f"Retrieved sources: {', '.join(result.get('sources', [])) or '(none)'}", flush=True)
        ticket = result.get("ticket_result") or {}
        if ticket:
            print(f"Ticket created: {ticket.get('created', False)}", flush=True)
        if result.get("audio_error"):
            print(f"Speech output error: {result['audio_error']}", flush=True)
        else:
            print(f"Spoken reply: {result['audio_reply']}", flush=True)
            if not args.no_playback:
                play_audio(result["audio_reply"], "assistant reply")

    report_path = output_dir / "demo_conversation.json"
    report_path.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nConversation record: {report_path}", flush=True)


if __name__ == "__main__":
    main()
