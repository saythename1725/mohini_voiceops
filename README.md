# VoiceOps Agent

The runnable application has one mode: the final RAG + structured-tools support
agent. The attached capstone brief and supplied dataset define its behavior.

## Install and run

Use Python 3.10 or newer. From this folder:

```text
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama serve
```

Copy `.env.example` to `.env`. Set `GEMINI_API_KEY` for retrieval embeddings,
`VOICEOPS_DATASET_ROOT` to the separately supplied dataset folder containing `kb/`,
`data/`, and `tools/`, and `GNANI_API_KEY` for voice mode. Keep Ollama running in
another terminal; it handles query rewriting and answer generation. Retrieval
requires Gemini embeddings with `EMBEDDING_MODEL=gemini-embedding-001`; if Gemini is
unavailable, retrieval reports an error and does not switch to keyword search. To run the full practice
questions and conversation suite:

```text
python -m src.practice_eval
```

The JSON report is saved to `outputs/practice_report.json`; it includes accuracy,
per-case results, replies, evidence section IDs, and actual/expected tool calls. To
avoid losing completed work, progress is checkpointed after every question and
conversation turn in `outputs/practice_progress.json`. An interrupted run resumes
automatically when you rerun the same command; you can also pass `--resume` explicitly.
For spreadsheet analysis, `outputs/practice_progress.csv` is rewritten after each
case with one row per question or conversation turn. It includes pass/fail, input,
reply, failure reasons, evidence, tool checks, language, and model usage. The JSON
checkpoint stays nested and keyed by IDs for reliable resume.
Use `--fresh` to discard the incomplete checkpoint and start the evaluation over.
If a conversation was interrupted partway through, that conversation is rerun from
its first turn so its in-memory state is reconstructed correctly; completed questions
and completed conversations are skipped.

## Reproducible evaluation

To keep baseline and candidate runs separate, give each run its own report and
checkpoint paths. Dense Gemini retrieval is the production baseline:

```text
python -m src.practice_eval --fresh --retrieval dense --output outputs/baseline_dense_practice_report.json --checkpoint outputs/baseline_dense_progress.json
```

The experimental hybrid variant fuses Gemini dense ranking with BM25 using RRF; it
still requires Gemini and is never used as a failure fallback. Use it for a separate
end-to-end run only if the retrieval comparison shows a benefit:

```text
python -m src.practice_eval --fresh --retrieval hybrid --output outputs/hybrid_practice_report.json --checkpoint outputs/hybrid_progress.json
```

For a retrieval-only comparison against the question set's gold evidence, run:

```text
python -m src.retrieval_eval --questions-only --output-dir outputs/evaluation/retrieval
```

After applying candidate changes, run the improved dense agent into a separate report:

```text
python -m src.practice_eval --fresh --retrieval dense --output outputs/improved_dense_practice_report.json --checkpoint outputs/improved_dense_progress.json
```

Once both agent reports and the retrieval comparison exist, generate the 13-field
JSONL results, diagnostics, summaries, retrieval rankings, failure CSV, and Markdown
report:

```text
python -m src.evaluation_artifacts --baseline outputs/baseline_dense_practice_report.json --improved outputs/improved_dense_practice_report.json --retrieval-comparison outputs/evaluation/retrieval/retrieval_comparison.json --output-dir outputs/evaluation
```

To evaluate the supplied transcript-like ASR variants separately, run:

```text
python -m src.practice_eval --fresh --asr-text --questions-only --output outputs/asr_text_practice_report.json --checkpoint outputs/asr_text_progress.json
```

Then add `--asr-report outputs/asr_text_practice_report.json` to the artifact command.

The evaluator uses annotations only after each agent response. The question set does
not contain expected tool calls, so tool accuracy is reported only for annotated
conversation turns. These ASR-like text variants assess downstream behavior on text,
not audio recognition quality.

Run the offline regression suite with:

```text
python -m unittest discover -s tests -v
```

The agent detects Hindi and Hinglish per turn, gives a language-specific response
instruction, and the evaluator flags Hindi replies that are not in Devanagari. When
Gemini retrieval or answer generation is unavailable, it gives a localized uncertainty
response and records the provider error instead of quoting an unrelated KB section. Retrieval
always includes the KB precedence rule (`KB-00.1`); dated electronics and return-window
cases also include the relevant current/historical policy sections.

To run one case, for example the first row in the practice file:

```text
python -m src.practice_eval --question-id PQ0002 --questions-only
```

For speech-like `asr_text` variants, add `--asr-text`.

## Submission package

The submission ZIP is organized with `README.md`, `.env.example`,
`requirements.txt`, `run.py`, `run.sh`, `src/`, `report.pdf`, `demo.mp4`, and
`sample_outputs/`. The supplied dataset and real `.env` are kept outside the ZIP.
Set `VOICEOPS_DATASET_ROOT` to the provided dataset folder before running; the
folder must contain `kb/`, `data/`, and `tools/`.

Repository URL: add the public GitHub URL after the repository is published.

Exact batch audio command (use your own audio folder and locale):

```text
python run.py path\to\audio_inputs --voice --language hi-IN
```

The ZIP's `demo.mp4` is a short walkthrough using an actual saved agent response
and the implemented voice architecture. For a live audio run, use the command above.

For text inputs, put one customer's message in each `.txt` file and run:

```text
python run.py path\to\text_inputs
```

Each file is treated as an independent customer. JSON answers and sources are saved
in `outputs/`; `run.bat` runs files in `text_inputs` on Windows.

## Final agent architecture

Markdown is split at section headings, retaining section IDs and YAML metadata.
Gemini embeddings are indexed in persistent Chroma at `outputs/chroma_db/`; the
embedding cache speeds index creation and is keyed by model and corpus. The first
run with a new embedding model re-embeds the KB and creates a separate Chroma
collection. Retrieval prioritizes current policy, then agent guide, help, FAQ, and
marketing. Historical support logs are not treated as policy. Before retrieval,
Ollama rewrites each query using current KB section IDs and titles while retaining
the original wording for search and answer grounding. Customer-facing policy answers
are generated by the LLM from retrieved evidence; topic-specific canned answers are
not used. Each answer stores the retrieved section IDs.

Pandas loads the CSV tables with `dtype=str` and `keep_default_na=False`, preserving
IDs and pincodes. The data layer joins orders, customers, order items/products, and
delivery pincode rows. The supplied `tools/tools.py` helpers are used for order
verification and support-ticket creation. Private order details are withheld until
phone-suffix or delivery-pincode verification succeeds. A ticket is only reported as
created if the helper returns `created=True`. Conversation history, the current issue,
verification state, ticket state, and prior tool results are retained within a
conversation; verification values are redacted from tool logs and ticket summaries.

The dataset's fixed reference date is 20 October 2026. Business-day deadlines exclude
Sundays and dates in `data/holidays.csv`. If retrieved evidence does not answer the
question, the assistant should say so rather than invent policy.

The measured no-RAG baseline is retained as a historical benchmark in
`outputs/baseline_report.json` (9/52, 17.31%). It is not an alternate runnable mode.

## Voice batch pipeline

The `--voice` path sends each recording in an audio folder through Gnani speech
recognition, the same RAG + tools support agent, and Gnani text-to-speech. It accepts
WAV, MP3, FLAC, OGG, and M4A recordings (Gnani REST STT supports files up to 60 seconds).
Copy `.env.example` to `.env` and set your own `GNANI_API_KEY`. For Hindi audio, choose
`hi-IN`; for English, choose `en-IN`:

```text
python run.py path\to\audio_inputs --voice --language hi-IN
```

For every supported audio file, `outputs/voice/` contains a transcript `.txt`, a
spoken reply `.wav`, and a JSON record with the transcript, text answer, retrieved KB
sources, tool calls, decision, and output paths. Each file is handled independently,
so one invalid recording does not stop the rest of the folder. Never commit `.env` or
share your API key.

For a **multi-turn demo conversation**, put sequential clips in `audio_inputs/` with
numbered names such as `01_policy.wav`, `02_followup.wav`, and `03_handover.wav`,
then run this from `voiceops_agent/`:

```text
python run_voice_demo.py audio_inputs --language en-IN
```

This demo runner keeps one agent instance alive across the clips so follow-ups retain
conversation state. It processes clips in filename order and saves a combined record
to `outputs/voice/demo_conversation.json`, along with per-turn transcripts and speech.
On Windows it plays WAV clips directly and uses `ffplay` for other input formats,
letting a screen recording capture both sides of the conversation. Install FFmpeg and
add `ffplay` to `PATH` to play non-WAV clips; they are still transcribed without it.
Add `--no-playback` to disable playback.
