"""Markdown section parser and persistent Chroma vector retrieval for the KB."""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class Chunk:
    source: str
    section_id: str
    title: str
    text: str
    metadata: dict[str, str]


def _bm25_tokens(text: str) -> list[str]:
    # Include Unicode letters/digits so Hindi queries do not become empty tokens.
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


class KnowledgeBase:
    def __init__(self, kb_dir: str | Path):
        self.kb_dir = Path(kb_dir)
        self.verbose = False
        self.chunks = self._load()
        self._query_vectors: dict[tuple[str, str], list[float]] = {}

    def _load(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path in sorted(self.kb_dir.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            metadata: dict[str, str] = {}
            body_lines: list[str] = []
            lines = raw.splitlines()
            if lines and lines[0].strip() == "---":
                end_frontmatter = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), 0)
                frontmatter = lines[1:end_frontmatter]
                body_lines = lines[end_frontmatter + 1:] if end_frontmatter else lines
            else:
                frontmatter = []
                body_lines = lines
            for line in frontmatter:
                match = re.match(r"^([a-z_]+):\s*(.*)$", line)
                if match:
                    metadata[match.group(1)] = match.group(2).strip()
            body = "\n".join(body_lines)
            headings = list(re.finditer(r"^##\s*\[([^]]+)\]\s*(.*)$", body, re.M))
            for index, heading in enumerate(headings):
                end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
                text = body[heading.end():end].strip()
                chunks.append(Chunk(path.name, heading.group(1), heading.group(2).strip(), text, metadata))
        return chunks

    def search(self, query: str, limit: int = 5) -> list[Chunk]:
        """Retrieve exclusively with Gemini embeddings; fail explicitly if unavailable."""
        if os.getenv("KB_EMBEDDINGS", "1").lower() in {"0", "false", "no"}:
            raise RuntimeError(
                "Gemini retrieval is required; KB_EMBEDDINGS cannot disable embeddings."
            )
        return self._semantic_search(query, limit)

    def search_hybrid(self, query: str, limit: int = 5, rrf_k: int = 60) -> list[Chunk]:
        """Experimental Gemini + BM25 retrieval for offline evaluation, using RRF."""
        if os.getenv("KB_EMBEDDINGS", "1").lower() in {"0", "false", "no"}:
            raise RuntimeError("Hybrid evaluation requires Gemini embeddings; embeddings are disabled.")
        candidate_count = min(len(self.chunks), max(20, limit * 5))
        dense = self._semantic_search(query, candidate_count)
        sparse = self._bm25_search(query, candidate_count)
        fused: dict[str, tuple[float, Chunk]] = {}
        for ranking in (dense, sparse):
            for rank, chunk in enumerate(ranking, start=1):
                key = f"{chunk.source}:{chunk.section_id}"
                score, _ = fused.get(key, (0.0, chunk))
                fused[key] = (score + 1.0 / (rrf_k + rank), chunk)
        ordered = sorted(
            fused.values(),
            key=lambda item: (item[0] + self._authority_boost(item[1]) * 1e-3,
                              -self._authority_rank(item[1])),
            reverse=True,
        )
        return [chunk for _, chunk in ordered[:limit]]

    def _semantic_search(self, query: str, limit: int) -> list[Chunk]:
        from .llm import GeminiEmbeddingClient
        import chromadb

        client = GeminiEmbeddingClient()
        if not self.chunks:
            return []
        model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        corpus = [f"{chunk.title}\n{chunk.text}" for chunk in self.chunks]
        fingerprint = hashlib.sha256((model + "\nRETRIEVAL_DOCUMENT\n" + "\n".join(corpus)).encode("utf-8")).hexdigest()
        safe_model = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")[:20]
        collection_name = f"zentromart_{safe_model}_{fingerprint[:16]}"
        store_path = Path(__file__).resolve().parents[1] / "outputs" / "chroma_db"
        chroma = chromadb.PersistentClient(path=str(store_path))
        collection = chroma.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine", "model": model,
                                            "corpus_fingerprint": fingerprint})
        ids = [f"{chunk.source}::{chunk.section_id}" for chunk in self.chunks]
        indexed = set(collection.get(include=["metadatas"]).get("ids", []))
        if not set(ids).issubset(indexed):
            vectors = self._load_or_build_vectors(client)
            metadata = []
            for chunk in self.chunks:
                metadata.append({"source": chunk.source, "section_id": chunk.section_id,
                                 "title": chunk.title, **chunk.metadata})
            collection.upsert(ids=ids, embeddings=vectors, documents=corpus, metadatas=metadata)
            if self.verbose:
                print(f"[rag] indexed {len(ids)} sections in persistent Chroma at {store_path}", flush=True)
        cache_key = (model, query)
        query_vector = self._query_vectors.get(cache_key)
        if query_vector is None:
            query_vector = client.embed([query], model=model, task_type="RETRIEVAL_QUERY")[0]
            self._query_vectors[cache_key] = query_vector
        candidate_count = min(len(self.chunks), max(20, limit * 5))
        result = collection.query(query_embeddings=[query_vector], n_results=candidate_count,
                                  include=["distances"])
        distance_by_id = dict(zip(result["ids"][0], result["distances"][0]))
        scored = []
        for chunk, chunk_id in zip(self.chunks, ids):
            if chunk_id not in distance_by_id:
                continue
            similarity = 1.0 - float(distance_by_id[chunk_id])
            scored.append((similarity + self._authority_boost(chunk), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:limit]]

    def _load_or_build_vectors(self, client):
        model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        corpus = [f"{chunk.title}\n{chunk.text}" for chunk in self.chunks]
        fingerprint = hashlib.sha256((model + "\nRETRIEVAL_DOCUMENT\n" + "\n".join(corpus)).encode("utf-8")).hexdigest()
        cache_path = Path(__file__).resolve().parents[1] / "outputs" / "kb_embeddings.json"
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("model") == model and cached.get("fingerprint") == fingerprint:
                if self.verbose:
                    print(f"[rag] loaded cached embeddings for {len(corpus)} sections", flush=True)
                return cached["vectors"]
        except (OSError, json.JSONDecodeError, TypeError):
            pass
        vectors = []
        batch_size = 12
        for start in range(0, len(corpus), batch_size):
            end = min(start + batch_size, len(corpus))
            if self.verbose:
                print(f"[rag] embedding KB sections {start + 1}-{end}/{len(corpus)}", flush=True)
            vectors.extend(client.embed(corpus[start:end], model=model, task_type="RETRIEVAL_DOCUMENT"))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"model": model, "fingerprint": fingerprint,
                                          "vectors": vectors}), encoding="utf-8")
        if self.verbose:
            print(f"[rag] cached embeddings at {cache_path}", flush=True)
        return vectors

    @staticmethod
    def _authority_rank(chunk: Chunk) -> int:
        rank = {"policy": 5, "agent_guide": 4, "help": 3, "faq": 2, "marketing": 1, "meta": 2}
        value = rank.get(chunk.metadata.get("doc_type", ""), 0)
        status = chunk.metadata.get("status", "")
        return value + (10 if status == "current" else -10 if status == "superseded" else 0)

    @staticmethod
    def _authority_boost(chunk: Chunk) -> float:
        rank = {"policy": 5, "agent_guide": 4, "help": 3, "faq": 2, "marketing": 1, "meta": 2}
        score = rank.get(chunk.metadata.get("doc_type", ""), 0) * 0.008
        if chunk.metadata.get("status") == "current":
            score += 0.02
        elif chunk.metadata.get("status") == "superseded":
            score -= 0.15
        return score

    def _bm25_search(self, query: str, limit: int = 5) -> list[Chunk]:
        """Rank all KB sections with Okapi BM25; used only by hybrid evaluation."""
        terms = _bm25_tokens(query)
        if not terms or not self.chunks:
            return []
        documents = [_bm25_tokens(f"{chunk.title} {chunk.text}") for chunk in self.chunks]
        lengths = [len(document) for document in documents]
        average_length = sum(lengths) / max(1, len(lengths))
        document_frequency: dict[str, int] = {}
        for document in documents:
            for term in set(document):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        total = len(documents)
        k1, b = 1.5, 0.75
        ranked: list[tuple[float, Chunk]] = []
        for chunk, document, length in zip(self.chunks, documents, lengths):
            frequencies: dict[str, int] = {}
            for term in document:
                frequencies[term] = frequencies.get(term, 0) + 1
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                inverse_frequency = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (1.0 - b + b * length / max(1.0, average_length))
                score += inverse_frequency * frequency * (k1 + 1.0) / denominator
            if score:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: (item[0], self._authority_rank(item[1])), reverse=True)
        return [chunk for _, chunk in ranked[:limit]]
