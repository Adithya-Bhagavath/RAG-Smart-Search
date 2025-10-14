import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib import robotparser
import json
import os
import random
import time

# ------------------ Smart User-Agent Pool ------------------
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]


# ------------------ Helper: Domain Validator ------------------
def is_valid_url(url, base_domain):
    """Ensure the URL belongs to the same domain."""
    parsed = urlparse(url)
    return parsed.netloc.endswith(base_domain) and parsed.scheme in ["http", "https"]


# ------------------ Helper: Robots.txt Check ------------------
def is_allowed_by_robots(url):
    """Check robots.txt for crawling permissions."""
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/robots_log.txt"

    try:
        rp.set_url(base_url)
        rp.read()
        allowed = rp.can_fetch("*", url)
        if not allowed:
            print(f"🚫 Disallowed by robots.txt: {url}")
            with open(log_path, "a") as log:
                log.write(f"[BLOCKED] {url}\n")
        else:
            print(f"✅ Allowed by robots.txt: {url}")
        return allowed
    except Exception as e:
        print(f"⚠️ Could not read robots.txt for {base_url}: {e}")
        print(f"🚫 Defaulting to disallow {url} (safety first).")
        with open(log_path, "a") as log:
            log.write(f"[FAILED TO READ robots.txt] {url}\n")
        return False


# ------------------ Helper: Query-Aware Ranking ------------------
def rank_text_by_query(text, query):
    """Ranks text chunks based on keyword overlap with query."""
    if not query:
        return text
    paragraphs = [p.strip() for p in text.split(".") if len(p.split()) > 6]
    query_words = set(query.lower().split())
    ranked = sorted(
        paragraphs,
        key=lambda p: len(set(p.lower().split()) & query_words),
        reverse=True,
    )
    return ". ".join(ranked[:6])  # top 6 relevant sections


# ------------------ Async HTML Fetcher ------------------
async def fetch_page(session, url, sem):
    """Fetch a webpage asynchronously."""
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
    }
    async with sem:
        try:
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status == 200 and "text/html" in resp.headers.get("Content-Type", ""):
                    return await resp.text()
        except Exception as e:
            print(f"❌ Failed to fetch {url}: {e}")
            return None
    return None


# ------------------ HTML Cleaner ------------------
def clean_text(html):
    """Removes junk tags and extracts readable content."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find("body") or soup
    text_parts = []
    for tag in main.find_all(["h1", "h2", "h3", "p", "li", "div", "span"]):
        content = tag.get_text(strip=True)
        if len(content.split()) > 5 and not content.startswith("©"):
            text_parts.append(content)
    return " ".join(text_parts)


# ------------------ Extract Internal Links ------------------
def extract_links(html, base_url, base_domain):
    """Extracts links within the same domain."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        full_url = urljoin(base_url, href)
        domain = urlparse(full_url).netloc
        if base_domain in domain:
            links.add(full_url)
    return list(links)


# ------------------ Main Async Crawler ------------------
async def crawl_async(start_url, max_pages=50, max_depth=2, query=""):
    parsed = urlparse(start_url)
    base_domain = parsed.netloc
    visited, pages, blocked = set(), [], []
    queue = [(start_url, 0)]
    sem = asyncio.Semaphore(5)  # concurrency limit

    print(f"🌐 Starting async crawl from: {start_url}")
    print(f"📍 Restricting to domain: {base_domain}\n")

    async with aiohttp.ClientSession() as session:
        while queue and len(pages) < max_pages:
            url, depth = queue.pop(0)
            if url in visited or depth > max_depth:
                continue

            # Respect robots.txt
            if not is_allowed_by_robots(url):
                blocked.append(url)
                continue

            html = await fetch_page(session, url, sem)
            visited.add(url)
            if not html:
                continue

            text = clean_text(html)
            if text:
                ranked_text = rank_text_by_query(text, query)
                pages.append({"url": url, "content": ranked_text})
                print(f"✅ Crawled: {url} ({len(ranked_text)} chars)")

                # Early exit if query hits found
                relevant_hits = sum(q in ranked_text.lower() for q in query.lower().split())
                if relevant_hits >= 2:
                    print("🧠 Early stop — sufficient relevant hits found.")
                    break

                # Crawl deeper
                if depth < max_depth:
                    links = extract_links(html, url, base_domain)
                    for link in links:
                        if link not in visited and len(queue) < max_pages and is_valid_url(link, base_domain):
                            queue.append((link, depth + 1))
            await asyncio.sleep(0.2)  # polite delay

    # Wikipedia fallback
    if not pages:
        brand = base_domain.replace("www.", "").split(".")[0]
        fallback = f"https://en.wikipedia.org/wiki/{brand.capitalize()}"
        print(f"⚠️ No crawlable data found — using Wikipedia fallback → {fallback}")
        async with aiohttp.ClientSession() as session:
            html = await fetch_page(session, fallback, sem)
            if html:
                text = clean_text(html)
                ranked_text = rank_text_by_query(text, query)
                pages.append({"url": fallback, "content": ranked_text})

    print(f"\n🔍 Crawl complete. Collected {len(pages)} pages. Blocked: {len(blocked)} URLs.\n")

    os.makedirs("data", exist_ok=True)
    timestamp = int(time.time())
    save_path = f"data/crawled_{base_domain}_{timestamp}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    print(f"💾 Data saved to: {save_path}\n")

    return pages, blocked


# ------------------ Sync Wrapper for Flask ------------------
def crawl_website(start_url, max_pages=10, max_depth=2, query=""):
    """Synchronous Flask-compatible wrapper."""
    return asyncio.run(crawl_async(start_url, max_pages, max_depth, query))


# ------------------ Debug Mode ------------------
if __name__ == "__main__":
    asyncio.run(
        crawl_async(
            "https://www.python.org/",
            max_pages=5,
            max_depth=2,
            query="What is Python used for?"
        )
    )