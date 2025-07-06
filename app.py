import os
import time
import tracemalloc


import streamlit as st


from pathlib import Path
from prometheus_client import CollectorRegistry, Counter, Histogram, start_http_server
from semantic_search_engine.engine import SearchEngine
from semantic_search_engine.logger import get_logger
from semantic_search_engine.settings import Settings


tracemalloc.start()


logger = get_logger(__name__)


@st.cache_resource
def get_engine():
    return SearchEngine()

engine = get_engine()


st.markdown(
    """
    <style>
    .result-card { margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #eee; }
    .result-title a { font-size: 1.25rem; color: #1a0dab; text-decoration: none; }
    .result-title a:hover { text-decoration: underline; }
    .result-url { font-size: 0.875rem; color: #006621; margin-bottom: 0.25rem; }
    .result-snippet { font-size: 1rem; color: #545454; line-height: 1.4; }
    </style>
    """,
    unsafe_allow_html=True,
)


METRICS_REGISTRY = CollectorRegistry()

try:
    start_http_server(Settings.PROMETHEUS_PORT)
except OSError as e:
    if e.errno == 98:
        print(f"[Warning] Port {Settings.PROMETHEUS_PORT} already in use. Skipping Prometheus server.")
    else:
        raise
REQ_COUNT = Counter(
    "search_requests_total",
    "Total number of search requests",
    registry=METRICS_REGISTRY
)
REQ_LAT = Histogram(
    "search_request_latency_seconds",
    "Histogram of search request latencies",
    registry=METRICS_REGISTRY
)

st.set_page_config(page_title="Semantic Search", layout="wide")
st.title("Semantic Search Engine")


@st.cache_data(ttl=3600)
def do_ingest(uploaded_file, url):
    sources = []
    if uploaded_file is not None:
        path = Settings.DATA_DIR / uploaded_file.name
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        sources.append(str(path))
    if url:
        sources.append(url)

    engine.ingest(sources)
    return True


with st.sidebar:
    st.header("Ingest Documents")
    up_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    url_in = st.text_input("Or enter a URL")
    if st.button("Start Ingestion"):
        if do_ingest(up_file, url_in):
            st.success("Ingestion completed!")


st.header("Search")
query = st.text_input("Enter your query here…")
if st.button("Search") and query:
    REQ_COUNT.inc()
    start = time.time()
    raw_hits = engine.search(query)
    REQ_LAT.observe(time.time() - start)

    if not raw_hits:
        st.warning("No results found.")
    else:
        # Dedupe by source, keep highest‐scoring first
        seen, unique = set(), []
        for hit in raw_hits:
            src = hit["meta"].get("source") or hit.get("source", "")
            if src in seen: 
                continue
            seen.add(src)
            unique.append(hit)
            if len(unique) >= Settings.TOP_K:
                break

        # Render like Google
        for hit in unique:
            src = hit["meta"].get("source") or hit.get("source", "")
            title = os.path.basename(src) or src
            text = hit.get("text", "")
            # Build a snippet around the query
            idx = text.lower().find(query.lower())
            if idx != -1:
                start_snip = max(0, idx - 80)
                end_snip   = min(len(text), idx + len(query) + 80)
                snippet = text[start_snip:end_snip]
                if start_snip > 0:    snippet = "…" + snippet
                if end_snip < len(text): snippet += "…"
            else:
                snippet = text[:160] + ("…" if len(text) > 160 else "")

            # Highlight the query
            snippet = snippet.replace(
                query, f"<strong>{query}</strong>"
            )

            st.markdown(
                f"""
                <div class="result-card">
                  <div class="result-title">
                    <a href="{src}" target="_blank">{title}</a>
                  </div>
                  <div class="result-url">{src}</div>
                  <div class="result-snippet">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )