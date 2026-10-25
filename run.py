"""Run the agent on text files or, with Gnani configured, audio files."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from src.agent import SupportAgent
from src.voice import SUPPORTED_AUDIO_EXTENSIONS, voice_turn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Folder containing .txt inputs, or audio files when --voice is enabled")
    parser.add_argument("--voice", action="store_true", help="Run the Gnani STT, RAG agent, and TTS pipeline on audio files")
    parser.add_argument("--language", default="en-IN", help="Gnani locale, for example en-IN, hi-IN, or gu-IN")
    args = parser.parse_args()
    input_folder = Path(args.input_folder)
    output_folder = Path(__file__).parent / "outputs"
    output_folder.mkdir(exist_ok=True)
    if not input_folder.is_dir():
        parser.error(f"input folder does not exist: {input_folder}")
    paths = sorted(path for path in input_folder.iterdir() if path.is_file())
    audio_output = output_folder / "voice"
    audio_paths = [path for path in paths if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS]
    for index, path in enumerate(paths, start=1):
        try:
            if args.voice and path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                continue
            if not args.voice and path.suffix.lower() != ".txt":
                continue
            print(f"[{index}/{len(paths)}] processing {path.name}", flush=True)
            agent = SupportAgent(use_llm=True, verbose=True)
            if not args.voice:
                result = agent.respond(path.read_text(encoding="utf-8"))
            else:
                result = voice_turn(path, agent, audio_output, language_code=args.language)
            target_folder = audio_output if args.voice else output_folder
            (target_folder / f"{path.stem}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{path.name}: {result['reply']}", flush=True)
            if result.get("audio_error"):
                print(f"{path.name}: TTS ERROR: {result['audio_error']}", flush=True)
        except Exception as exc:
            print(f"{path.name}: ERROR: {exc}", flush=True)
    if args.voice:
        print(f"Audio files found: {len(audio_paths)}. Voice outputs: {audio_output}", flush=True)


if __name__ == "__main__":
    main()
