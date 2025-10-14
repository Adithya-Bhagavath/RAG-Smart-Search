import os
import re
import nltk
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline

# Ensure NLTK sentence tokenizer is ready
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class AsyncSummarizer:
    def __init__(self):
        """
        ⚡ GPU-Aware Async Query-Aware Summarizer
        Uses DistilBART and runs chunks concurrently.
        """
        model_name = os.getenv("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")

        # 🔍 Auto-detect GPU availability
        device = 0 if torch.cuda.is_available() else -1
        device_type = "GPU" if device == 0 else "CPU"
        print(f"🧠 Loading summarizer model ({model_name}) on {device_type}...")

        # Load summarization pipeline
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=device
        )

        # Thread pool for async execution (even GPU ops)
        self.executor = ThreadPoolExecutor(max_workers=4)

    # -------------------------------------------------------
    # 🧹 Text Cleanup Helpers
    # -------------------------------------------------------
    def _clean_text(self, text):
        """Remove extra whitespace and citations."""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\[[0-9]+\]', '', text)
        return text

    def _chunk_text(self, text, max_len=3000):
        """Split text into smaller segments."""
        if len(text) <= max_len:
            return [text]

        sentences = nltk.sent_tokenize(text)
        chunks, current = [], ""

        for sent in sentences:
            if len(current) + len(sent) < max_len:
                current += " " + sent
            else:
                chunks.append(current.strip())
                current = sent
        if current:
            chunks.append(current.strip())
        return chunks

    def _deduplicate_sentences(self, text):
        """Eliminate duplicate or near-identical sentences."""
        sentences = nltk.sent_tokenize(text)
        seen, filtered = set(), []
        for sent in sentences:
            cleaned = sent.lower().strip()
            if cleaned not in seen and len(cleaned) > 25:
                seen.add(cleaned)
                filtered.append(sent)
        return " ".join(filtered)

    # -------------------------------------------------------
    # ⚙️ Core Async Summarization Logic
    # -------------------------------------------------------
    async def _summarize_chunk(self, prompt, max_len, min_len):
        """Run a single chunk summarization asynchronously."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            lambda: self.summarizer(
                prompt,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
            )
        )
        return result[0]["summary_text"]

    # -------------------------------------------------------
    # 🧩 Main Async Summarization
    # -------------------------------------------------------
    async def summarize(self, text, query=None):
        """
        Summarizes text concurrently, aware of query and context.
        Auto-selects GPU if available.
        """
        if not text or not text.strip():
            return "No relevant content found."

        text = self._clean_text(text)
        text = self._deduplicate_sentences(text)
        chunks = self._chunk_text(text)

        print(f"🧩 Summarizing {len(chunks)} chunk(s) concurrently...")

        tasks = []
        for chunk in chunks:
            # Make summary more question-aware
            if query:
                prompt = (
                    f"Question: {query}\n\n"
                    f"Context: {chunk}\n\n"
                    f"Provide a concise answer based only on the context."
                )
            else:
                prompt = chunk

            max_len = min(200, max(60, len(prompt) // 25))
            min_len = max(30, max_len // 3)

            tasks.append(self._summarize_chunk(prompt, max_len, min_len))

        # Run summarization tasks concurrently
        summaries = await asyncio.gather(*tasks)
        combined_summary = " ".join(summaries).strip()

        # Filter summary by query relevance
        if query:
            query_terms = query.lower().split()
            relevant_sentences = [
                s for s in nltk.sent_tokenize(combined_summary)
                if any(q in s.lower() for q in query_terms)
            ]
            if relevant_sentences:
                combined_summary = " ".join(relevant_sentences)

        return combined_summary or "No clear answer could be derived."