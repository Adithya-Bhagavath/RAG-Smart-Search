from flask import Flask, render_template, request, jsonify
from crawler import crawl_async
from embeddings import EmbeddingEngine
from summarizer_async import AsyncSummarizer as Summarizer
import asyncio
import logging

# --------------------------------------------------
# 🔧 Setup & Initialization
# --------------------------------------------------
app = Flask(__name__, template_folder="templates")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Global engine + summarizer (persistent)
engine = EmbeddingEngine("sentence-transformers/all-MiniLM-L6-v2")
summarizer = Summarizer()

# --------------------------------------------------
# 🧠 Helper: Run concurrent crawls
# --------------------------------------------------
async def crawl_concurrently(urls, query):
    tasks = [
        crawl_async(url, max_pages=50, max_depth=2, query=query)
        for url in urls if url
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    all_pages, all_blocked = [], []
    for pages, blocked in results:
        all_pages.extend(pages)
        all_blocked.extend(blocked)
    return all_pages, all_blocked

# --------------------------------------------------
# 🏠 Homepage
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# --------------------------------------------------
# 🌐 Crawl Endpoint (SYNC Friendly)
# --------------------------------------------------
@app.route("/api/crawl", methods=["POST"])
def crawl():
    """Crawl domains WITHOUT using async/await directly."""
    data = request.get_json()
    url1 = data.get("url")
    url2 = data.get("url2")

    if not url1 and not url2:
        return jsonify({"success": False, "message": "Please provide at least one URL."}), 400

    print(f"🌐 Crawl initiated for: {url1 or url2}")

    # ✅ Run async concurrent crawl with asyncio.run
    all_pages, all_blocked = asyncio.run(crawl_concurrently([url1, url2], query=""))

    if not all_pages:
        return jsonify({
            "success": False,
            "message": "No pages found — possibly blocked by robots.txt.",
            "blocked": all_blocked
        }), 400

    # ✅ Run build_index in background thread
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(engine.build_index, all_pages)

    message = f"✅ Crawled {len(all_pages)} pages successfully!"
    if all_blocked:
        message += f" 🚫 {len(all_blocked)} URLs blocked by robots.txt."

    print(f"✅ Crawl completed — {len(all_pages)} pages, {len(all_blocked)} blocked.\n")

    return jsonify({
        "success": True,
        "pages": len(all_pages),
        "blocked": all_blocked,
        "message": message
    })

# --------------------------------------------------
# 🤖 Search Endpoint (KEEP ASYNC)
# --------------------------------------------------
@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    url1 = data.get("url", "").strip()
    url2 = data.get("url2", "").strip()
    smart = data.get("smart", False)

    if not query:
        return jsonify({"success": False, "message": "Query is required."}), 400

    print(f"🔍 Searching for: '{query}' across {url1 or 'N/A'} {url2 or ''}")

    all_pages, all_blocked = [], []

    # Crawl primary site
    if url1:
        print(f"🌐 Crawling primary: {url1}")
        pages1, blocked1 = asyncio.run(crawl_concurrently([url1], query=query))
        all_pages.extend(pages1)
        all_blocked.extend(blocked1)

    # Crawl secondary site
    if url2:
        print(f"🌐 Crawling secondary: {url2}")
        pages2, blocked2 = asyncio.run(crawl_concurrently([url2], query=query))
        all_pages.extend(pages2)
        all_blocked.extend(blocked2)

    # Fallback to Wikipedia if nothing found
    if not all_pages:
        print("⚠️ No data from domains — using Wikipedia fallback.")
        fallback, blocked_fallback = asyncio.run(
            crawl_async("https://en.wikipedia.org/wiki/Main_Page", max_pages=2)
        )
        all_pages.extend(fallback)
        all_blocked.extend(blocked_fallback)

    # Still nothing to search
    if not all_pages:
        return jsonify({
            "success": False,
            "message": "No readable content found.",
            "blocked": all_blocked
        }), 404

    # Build embeddings
    global engine
    engine.build_index(all_pages)

    # ---- SAFE HYBRID SEARCH ----
    results = []  # always defined
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(engine.search, query, 7, 0.7)
            results = future.result()
    except Exception as e:
        print(f"❌ Search error: {e}")
        return jsonify({
            "success": False,
            "message": f"Internal search error: {str(e)}",
            "blocked": all_blocked
        }), 500

    # Handle no results
    if not results:
        return jsonify({
            "success": True,
            "summary": "❌ No relevant information found.",
            "results": [],
            "blocked": all_blocked
        })

    # ---- SMART SUMMARIZATION ----
    summary = None
    if smart:
        print("🧠 Smart Mode enabled — summarizing asynchronously...")
        combined_text = " ".join([r["content"] for r in results])
        try:
            summary = asyncio.run(summarizer.summarize(combined_text, query))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            summary = loop.run_until_complete(summarizer.summarize(combined_text, query))
        except Exception as e:
            print(f"⚠️ Summarizer error: {e}")
            summary = "Unable to summarize content."

    print(f"✅ Search complete for '{query}' — {len(results)} results.\n")
    return jsonify({
        "success": True,
        "summary": summary,
        "results": results,
        "blocked": all_blocked
    })

# --------------------------------------------------
# 🚀 Main Runner
# --------------------------------------------------
if __name__ == "__main__":
    print("🚀 Launching Konduit Smart Search (Async Mode)...")
    app.run(host="0.0.0.0", port=8000, debug=True)