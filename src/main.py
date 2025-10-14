import os
from dotenv import load_dotenv
import warnings
import textwrap
from crawler import crawl_website
from embeddings import EmbeddingEngine

# Load environment variables
load_dotenv()
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

if __name__ == "__main__":
    start_url = input("🔗 Enter website URL to crawl: ").strip()
    if not start_url.startswith("http"):
        start_url = "https://" + start_url

    limit = int(os.getenv("CRAWL_LIMIT", 10))

    print(f"🌐 Crawling started from: {start_url}\n")
    pages = crawl_website(start_url, max_pages=limit)

    if not pages:
        print("❌ No pages crawled. Exiting.")
        exit()

    print(f"\n💾 Crawled {len(pages)} pages. Building embeddings...\n")

    # Step 2: Build embeddings
    model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    engine = EmbeddingEngine(model_name)
    engine.build_index(pages)
    print("✅ Index built successfully!")

    # Step 3: Query Loop
    # Step 3: Query Loop
    while True:
        query = input("\n💬 Enter your search query (or type 'exit'): ").strip()
        if query.lower() == "exit":
            print("👋 Exiting.")
            break

        results = engine.search(query, top_k=3)
        print("\n🤖 Smart Answers (Top Results):\n" + "=" * 60)

        for i, r in enumerate(results, 1):
            print(f"\n📘 Result {i}")
            print(f"🌐 URL: {r['url']}")
            print(f"🔢 Similarity Score: {r['score']:.3f}")
            print(f"🧠 Answer:\n{r['content']}")
            print("-" * 60)