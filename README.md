# 🧠 Smart Search  
**AI-Powered Asynchronous Web Crawler, Hybrid Search, and Query-Aware Summarizer**

---

## 🌟 Overview

Smart Search is an **AI-driven intelligent search system** that:
- Crawls 30–50 pages concurrently using asynchronous crawling (`aiohttp` + `asyncio`)
- Builds **semantic embeddings** using `SentenceTransformers`
- Performs **hybrid search** (semantic + keyword-based)
- Generates **query-aware summaries** via `DistilBART`

It’s an end-to-end pipeline that combines **speed**, **contextual intelligence**, and **ethical design**, built as part of the **Konduit SDE Intern Take-Home Assignment**.

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Backend Framework** | Flask 2.3 (Async Supported) |
| **Crawling** | aiohttp + asyncio + BeautifulSoup |
| **Embeddings Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Summarization Model** | `sshleifer/distilbart-cnn-12-6` |
| **Language Tools** | NLTK |
| **Frontend** | HTML + Vanilla JS |
| **Environment** | Python 3.9+ |

---

## 🧩 Key Features

✅ **Asynchronous Ethical Crawling**  
Crawls 30–50 pages concurrently, enforces robots.txt compliance, and rotates User-Agents.

✅ **Hybrid Semantic Search Engine**  
Combines transformer embeddings with keyword-based lexical matching.

✅ **Query-Aware Summarization**  
Answers user questions contextually, not generic summaries.

✅ **Multi-Domain Search**  
Accepts two URLs and performs concurrent crawling + search.

✅ **Wikipedia Fallback**  
Auto-fetches Wikipedia content if a site has no accessible data.

✅ **Front-End Dashboard**  
Interactive and minimalistic UI built in `templates/index.html`.

---

## 🚀 Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Akshay34-ux/konduit-smart-search.git
cd konduit-smart-search
```

### 2️⃣ Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run Flask Server
```bash
python src/app.py
```

Flask runs at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🌐 API Endpoints

### 🕷️ `POST /api/crawl`
Crawls one or two domains asynchronously and builds embeddings.

**Request Example**
```json
{
  "url": "https://www.python.org",
  "url2": "https://www.ibm.com"
}
```

**Response Example**
```json
{
  "success": true,
  "pages": 42,
  "blocked": ["https://www.ibm.com/admin/"],
  "message": "✅ Crawled 42 pages successfully!"
}
```

---

### 🔍 `POST /api/search`
Performs hybrid semantic + keyword-based search with optional AI summarization.

**Request Example**
```json
{
  "query": "What is Python used for?",
  "url": "https://www.python.org",
  "url2": "https://www.ibm.com",
  "smart": true
}
```

**Response Example**
```json
{
  "success": true,
  "summary": "Python is widely used in web development, data science, automation, and AI.",
  "results": [
    {
      "url": "https://www.python.org/about/",
      "semantic_score": 0.88,
      "keyword_score": 0.72,
      "final_score": 0.84
    }
  ],
  "blocked": []
}
```

---

### 💓 `GET /health`
Check server health.

**Response**
```json
{
  "status": "ok",
  "message": "Konduit Smart Search backend is running!"
}
```

---

## 📁 Directory Structure

```
konduit-smart-search/
│
├── src/
│   ├── app.py               # Flask backend (async)
│   ├── crawler.py           # Asynchronous ethical crawler
│   ├── embeddings.py        # Hybrid search engine
│   ├── summarizer_async.py  # Async summarizer (DistilBART)
│   └── templates/
│       └── index.html       # User interface
│
├── data/                    # Stored crawled data and embeddings
├── logs/                    # robots.txt logs
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---
                   ┌────────────────────────────────┐
                   │        User Interface (UI)     │
                   │  - Enter URLs + query          │
                   │  - Enable Smart Mode 🤖        git remote -v│
                   └──────────────┬─────────────────┘
                                  │
                                  ▼
                  ┌────────────────────────────────┐
                  │          Flask Backend         │
                  │  (app.py)                      │
                  └──────────────┬─────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │                 Async Crawler                    │
        │  - Crawls 30–50 pages concurrently               │
        │  - Checks robots.txt & cleans text               │
        │  - Extracts internal links                       │
        │  - Saves data → data/crawled_<domain>.json       │
        └────────────────────┬─────────────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────────────┐
        │             Embedding Engine (MiniLM)            │
        │  - Converts text chunks → vector embeddings      │
        │  - Stores → data/embeddings.json                 │
        └────────────────────┬─────────────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────────────┐
        │              Hybrid Search Engine                │
        │  - Combines Semantic + Keyword relevance         │
        │  - Ranks top results                             │
        └────────────────────┬─────────────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────────────┐
        │        AI Summarizer (DistilBART) 🤖             │
        │  - Generates concise, query-specific answers     │
        │  - Summarizes only top-ranked pages              │
        └────────────────────┬─────────────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────────────┐
        │              Response to Frontend                │
        │  - Smart Summary (if enabled)                    │
        │  - Top Sources + Scores                          │
        └──────────────────────────────────────────────────┘

## 👨‍💻 Author
**Akshay LN**  
_MCA Student | AI & Full Stack Developer_  
📧 [adibhagavath03@gmail.com](mailto:your.email@example.com)    
⭐ [GitHub](https://github.com/Adithya-Bhagavath)

---

## 🪪 License
**MIT License**  
You are free to use, modify, and distribute this project with attribution.

---

## 💡 Credits  
as part of the **Konduit SDE Intern Take-Home Assignment**.
