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
query = st.text_input("Enter your query here...")
if st.button("Search") and query:
    REQ_COUNT.inc()
    start = time.time()
    raw = engine.search(query)
    REQ_LAT.observe(time.time() - start)

    if not raw:
        st.warning("No results found.")
    else:
        seen = {}
        for hit in raw:
            src = hit["meta"].get("source") or hit.get("payload",{}).get("source")
            if src not in seen or hit.get("score",0) < seen[src].get("score", float("inf")):
                seen[src] = hit

        st.markdown("### Top hits by file")
        for src, hit in seen.items():
            text = hit.get("text", "")            
            idx = text.lower().find(query.lower())
            if idx != -1:
                start_snip = max(0, idx - 80)
                end_snip   = min(len(text), idx + len(query) + 80)
                snippet = text[start_snip:end_snip]
                if start_snip>0:    snippet = "…" + snippet
                if end_snip<len(text): snippet = snippet + "…"
            else:
                snippet = text[:160] + ("…" if len(text)>160 else "")

            st.write(f"**File:** `{src}`")
            highlighted = snippet.replace(
                query, f"**{query}**"
            )
            st.markdown(f"> {highlighted}")
            
            try:
                with open(src, "rb") as f:
                    data_bytes = f.read()
                st.download_button(
                    "Download file",
                    data_bytes,
                    file_name=src.split("/")[-1],
                    mime="application/octet-stream"
                )
            except FileNotFoundError:
                pass