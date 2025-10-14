from transformers import pipeline
import os
import re
import nltk

# Ensure sentence tokenizer is ready
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class Summarizer:
    def __init__(self):
        """
        Enhanced query-aware summarizer using DistilBART.
        Produces contextually focused summaries aligned with the user's question.
        """
        model_name = os.getenv("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")
        print(f"🧠 Loading summarizer model ({model_name})...")

        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=-1  # CPU-friendly (set to 0 for GPU)
        )

    # -------------------------------------------------------
    # 🧹 Clean Text
    # -------------------------------------------------------
    def _clean_text(self, text):
        """Removes extra whitespace and citations."""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\[[0-9]+\]', '', text)
        return text

    # -------------------------------------------------------
    # ✂️ Chunk Long Text
    # -------------------------------------------------------
    def _chunk_text(self, text, max_len=3000):
        """Splits very long content into smaller processable chunks."""
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

    # -------------------------------------------------------
    # 🧩 Deduplicate Similar Sentences
    # -------------------------------------------------------
    def _deduplicate_sentences(self, text):
        """Removes repeated or near-identical lines."""
        sentences = nltk.sent_tokenize(text)
        seen = set()
        filtered = []
        for sent in sentences:
            cleaned = sent.lower().strip()
            if cleaned not in seen and len(cleaned) > 25:
                seen.add(cleaned)
                filtered.append(sent)
        return " ".join(filtered)

    # -------------------------------------------------------
    # 💡 Query-Aware Summarization
    # -------------------------------------------------------
    def summarize(self, text, query=None):
        """
        Generates a focused, query-relevant summary instead of a generic one.
        """
        if not text or not text.strip():
            return "No relevant content found."

        text = self._clean_text(text)
        text = self._deduplicate_sentences(text)
        chunks = self._chunk_text(text)

        summaries = []
        print(f"🧩 Summarizing {len(chunks)} chunk(s)...")

        try:
            for chunk in chunks:
                # Make the model understand the question
                if query:
                    prompt = (
                        f"Question: {query}\n\n"
                        f"Context: {chunk}\n\n"
                        f"Answer the question above briefly and clearly "
                        f"based only on the context provided."
                    )
                else:
                    prompt = chunk

                max_len = min(200, max(60, len(prompt) // 25))
                min_len = max(30, max_len // 3)

                result = self.summarizer(
                    prompt,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                )
                summaries.append(result[0]["summary_text"])

            combined_summary = " ".join(summaries).strip()

            # If query provided, prioritize lines that actually answer it
            if query:
                query_terms = query.lower().split()
                relevant = [
                    s for s in nltk.sent_tokenize(combined_summary)
                    if any(q in s.lower() for q in query_terms)
                ]
                if relevant:
                    combined_summary = " ".join(relevant)

            return combined_summary or "No clear answer could be derived."

        except Exception as e:
            print(f"⚠️ Summarizer error: {e}")
            return "Unable to generate a meaningful summary."