import os
import io
import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from src.knowledge_base import Chunk, KnowledgeBase
from src.llm import GeminiEmbeddingClient


class GeminiOnlyRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase(os.path.join(os.path.dirname(__file__), "empty_kb"))
        self.keyword_called = False

        def keyword_search(*args, **kwargs):
            self.keyword_called = True
            return []

        self.kb._keyword_search = keyword_search

    def test_gemini_failure_does_not_switch_to_keyword_search(self):
        failure = RuntimeError("Gemini is unavailable")

        def semantic_search(*args, **kwargs):
            raise failure

        self.kb._semantic_search = semantic_search
        with patch.dict(os.environ, {"KB_EMBEDDINGS": "1"}):
            with self.assertRaisesRegex(RuntimeError, "Gemini is unavailable"):
                self.kb.search("return policy")
        self.assertFalse(self.keyword_called)

    def test_embeddings_cannot_be_disabled_for_retrieval(self):
        self.kb._semantic_search = lambda *args, **kwargs: self.fail(
            "Gemini search must not run when explicitly disabled"
        )
        with patch.dict(os.environ, {"KB_EMBEDDINGS": "0"}):
            with self.assertRaisesRegex(RuntimeError, "Gemini retrieval is required"):
                self.kb.search("return policy")
        self.assertFalse(self.keyword_called)

    def test_bm25_finds_a_relevant_policy_section(self):
        policy = Chunk("policy.md", "RET-2", "Return window", "Customers can request a return.",
                       {"doc_type": "policy", "status": "current"})
        unrelated = Chunk("faq.md", "FAQ-1", "Payment methods", "Pay by card or UPI.",
                          {"doc_type": "faq", "status": "current"})
        self.kb.chunks = [unrelated, policy]
        self.assertEqual(self.kb._bm25_search("return window", 1), [policy])

    def test_dense_search_never_returns_sections_absent_from_gemini_ranking(self):
        first = Chunk("first.md", "A", "A", "first", {"doc_type": "faq", "status": "current"})
        second = Chunk("second.md", "B", "B", "second", {"doc_type": "policy", "status": "current"})
        self.kb.chunks = [first, second]
        collection = MagicMock()
        collection.get.return_value = {"ids": ["first.md::A", "second.md::B"]}
        collection.query.return_value = {"ids": [["first.md::A"]], "distances": [[0.2]]}
        chroma = MagicMock()
        chroma.get_or_create_collection.return_value = collection
        with patch("src.llm.GeminiEmbeddingClient") as client_class, \
                patch("chromadb.PersistentClient", return_value=chroma):
            client_class.return_value.embed.return_value = [[0.1, 0.2]]
            result = self.kb._semantic_search("query", 2)
        self.assertEqual(result, [first])

    def test_rrf_hybrid_fuses_dense_and_bm25_rankings(self):
        a = Chunk("a.md", "A", "A", "", {"doc_type": "faq", "status": "current"})
        b = Chunk("b.md", "B", "B", "", {"doc_type": "policy", "status": "current"})
        c = Chunk("c.md", "C", "C", "", {"doc_type": "help", "status": "current"})
        self.kb.chunks = [a, b, c]
        self.kb._semantic_search = lambda query, limit: [a, b]
        self.kb._bm25_search = lambda query, limit: [b, c]
        with patch.dict(os.environ, {"KB_EMBEDDINGS": "1"}):
            result = self.kb.search_hybrid("query", limit=3)
        self.assertEqual([chunk.section_id for chunk in result], ["B", "A", "C"])

    def test_current_policy_precedes_superseded_policy(self):
        current = Chunk("current.md", "P1", "Policy", "",
                        {"doc_type": "policy", "status": "current"})
        old = Chunk("old.md", "P0", "Old policy", "",
                    {"doc_type": "policy", "status": "superseded"})
        self.assertGreater(self.kb._authority_rank(current), self.kb._authority_rank(old))

    def test_gemini_retries_transient_rate_limit_without_changing_provider(self):
        rate_limit = HTTPError("https://example.invalid", 429, "rate limited", None, None)
        response = io.BytesIO(json.dumps({"embeddings": [{"values": [0.1, 0.2]}]}).encode())
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), \
                patch("src.llm.urlopen", side_effect=[rate_limit, response]) as request, \
                patch("src.llm.time.sleep"):
            vectors = GeminiEmbeddingClient().embed(["query"], task_type="RETRIEVAL_QUERY")
        self.assertEqual(vectors, [[0.1, 0.2]])
        self.assertEqual(request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
