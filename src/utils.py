import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def crawl_website(start_url, max_pages=5):
    visited = set()
    to_visit = [start_url]
    pages = []

    print(f"🌐 Crawling started from: {start_url}\n")

    while to_visit and len(pages) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            pages.append({"url": url, "content": text})
            visited.add(url)

            # collect more links to crawl
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.startswith("http") and href not in visited:
                    to_visit.append(href)

            print(f"✅ Crawled: {url} ({len(text.split())} words)")
        except Exception as e:
            print(f"❌ Failed: {url} | Error: {e}")

    print(f"\n🔍 Crawling completed. {len(pages)} pages collected.\n")
    return pages