# ZentroMart VoiceOps


## Setup

1. Install Python 3.10 or newer and [Ollama](https://ollama.com/).
2. From this folder, install dependencies and download the model:

   ```bat
   python -m pip install -r requirements.txt
   ollama pull qwen2.5:3b
   ```

3. Copy `.env.example` to `.env`. Set `GEMINI_API_KEY`, `GNANI_API_KEY`, and
   `VOICEOPS_DATASET_ROOT` to the supplied dataset folder containing `kb`, `data`,
   and `tools`. Keep `.env` private.
4. Start Ollama in a separate terminal:

   ```bat
   ollama serve
   ```

## Run a folder of audio

Put the audio files in a folder. Supported formats: WAV, MP3, FLAC, OGG, and M4A.
From this folder, run:

```bat
python run.py path\to\audio_inputs --voice --language en-IN
```

Use `hi-IN` for Hindi audio. Transcripts, spoken replies, and retrieved source IDs
are saved under `outputs\voice\`.
