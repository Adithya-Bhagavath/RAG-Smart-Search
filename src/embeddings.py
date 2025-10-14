from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import json
import os
import re
import math

class EmbeddingEngine:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"🧠 Loading bi-encoder model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("🤖 Loading cross-encoder reranker (ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.embeddings = None
        self.text_chunks = []
        self.urls = []
        self.query_cache = {}

    # -------------------------------------------------------
    # 🧩 Text Chunking
    # -------------------------------------------------------
    def chunk_text(self, text, max_length=300):
        """Split text into small readable chunks."""
        text = re.sub(r'\s+', ' ', text).strip()
        if not text or len(text) < 50:
            return []

        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks, current_chunk = [], ""

        for sent in sentences:
            if len(current_chunk) + len(sent) < max_length:
                current_chunk += sent + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sent + " "
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # -------------------------------------------------------
    # 🧠 Build Vector Index
    # -------------------------------------------------------
    def build_index(self, pages):
        """Encode chunks and build embedding index."""
        print("⚙️ Building embeddings with semantic context...")
        self.text_chunks = []
        self.urls = []

        for page in pages:
            content = page.get("content", "").strip()
            if not content:
                continue
            chunks = self.chunk_text(content)
            if chunks:
                self.text_chunks.extend(chunks)
                self.urls.extend([page.get("url", "unknown")] * len(chunks))

        if len(self.text_chunks) == 0:
            print("⚠️ No valid text found. Skipping embedding build.")
            self.embeddings = None
            return

        self.embeddings = self.model.encode(self.text_chunks, convert_to_tensor=True, show_progress_bar=True)
        print(f"✅ Indexed {len(self.text_chunks)} chunks from {len(pages)} pages.")
        os.makedirs("data", exist_ok=True)
        with open("data/embeddings.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": self.text_chunks, "urls": self.urls}, f, ensure_ascii=False, indent=2)
        print("💾 Embeddings saved to data/embeddings.json")

    # -------------------------------------------------------
    # 🧮 Keyword Overlap
    # -------------------------------------------------------
    def _keyword_overlap(self, query, text):
        """Compute lexical overlap between query and text chunk."""
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        text_words = set(re.findall(r'\b\w{3,}\b', text.lower()))
        if not query_words or not text_words:
            return 0.0
        common = len(query_words & text_words)
        return round(common / math.sqrt(len(query_words) * len(text_words)), 3)

    # -------------------------------------------------------
    # 🔍 Hybrid Search (Semantic + Keyword)
    # -------------------------------------------------------
    def search(self, query, top_k=5, hybrid_weight=0.7, min_score=0.15):
        """Perform hybrid retrieval with reranking."""
        if self.embeddings is None or len(self.text_chunks) == 0:
            raise ValueError("❌ Embeddings not built yet. Run build_index() first.")

        print(f"\n🔎 Searching for: '{query}'")

        # Cache query embedding
        if query in self.query_cache:
            query_emb = self.query_cache[query]
        else:
            query_emb = self.model.encode(query, convert_to_tensor=True)
            self.query_cache[query] = query_emb

        # Semantic similarity
        semantic_scores = util.cos_sim(query_emb, self.embeddings)[0].detach().cpu()

        # Keyword overlap
        keyword_scores = torch.tensor(
            [self._keyword_overlap(query, chunk) for chunk in self.text_chunks],
            dtype=torch.float32
        )

        # Align shapes safely
        min_len = min(len(semantic_scores), len(keyword_scores))
        semantic_scores = semantic_scores[:min_len]
        keyword_scores = keyword_scores[:min_len]

        # Weighted combination
        hybrid_scores = (semantic_scores * hybrid_weight) + (keyword_scores * (1 - hybrid_weight))

        # Flatten and get top results
        hybrid_scores = hybrid_scores.flatten()
        top_results = torch.topk(hybrid_scores, k=min(top_k * 3, hybrid_scores.numel()))  # fetch more before rerank

        results = []
        for idx, score in zip(top_results.indices.tolist(), top_results.values.tolist()):
            if score >= min_score:
                results.append({
                    "url": self.urls[idx],
                    "content": self.text_chunks[idx],
                    "semantic_score": float(semantic_scores[idx]),
                    "keyword_score": float(keyword_scores[idx]),
                    "final_score": float(score)
                })

        print(f"✅ Retrieved {len(results)} candidate chunks before reranking.")

        # 🔁 Step 2: Rerank using CrossEncoder
        results = self.rerank_results(query, results)
        print(f"🏆 Top {len(results)} final chunks after reranking.\n")

        return results

    # -------------------------------------------------------
    # 🧩 Cross-Encoder Reranking
    # -------------------------------------------------------
    def rerank_results(self, query, results, top_k=5):
        """Use a cross-encoder to rerank top results based on relevance."""
        if not results:
            return results

        pairs = [(query, r["content"]) for r in results]
        scores = self.reranker.predict(pairs)

        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])
        results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        print(f"🧩 Re-ranked {len(results)} results using cross-encoder.")
        return results[:top_k]