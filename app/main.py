"""
Phase 5 - Interactive Streamlit Demo for GraphRAG Scientific Literature
Full-featured Q&A interface with query decomposition, PageRank reranking, and graph visualization.
"""

import streamlit as st
import json
import os
import sys
import urllib.request
import urllib.error

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="GraphRAG Scientific Literature Q&A",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FastAPI backend (Step 3: Streamlit talks to the API, never to Neo4j/LLM
# directly for the query flow). Configurable; defaults to local laptop API.
API_URL = os.getenv("GRAPHRAG_API_URL", "http://localhost:8000").rstrip("/")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from neo4j import GraphDatabase

from src.config import settings

from app.components.search_bar import render_search_bar, render_sidebar_settings
from app.components.answer_card import render_answer_card
from app.components.evidence_panel import render_evidence_panel
from app.components.graph_view import render_graph_view

# Load custom CSS
css_path = os.path.join(project_root, "app", "styles", "theme.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def init_graph_driver():
    """Read-only Neo4j driver used ONLY by the Knowledge Graph visualization.

    Retrieval and answer generation go through the FastAPI backend; this
    driver never runs retrieval queries, only the viz component's
    chunk/entity lookup.
    """
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        # Test connection
        with driver.session() as session:
            session.run("RETURN 1")
        return driver
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {str(e)}")
        st.info("Graph visualization needs Neo4j; answers still work via the API.")
        return None


def build_query_payload(query, system_mode, expand_graph, use_adaptive, use_gds,
                        use_decomposition, top_k) -> dict:
    """Build the POST /query body from the current UI settings (pure)."""
    return {
        "query": query,
        "system_mode": system_mode,
        "expand_graph": expand_graph,
        "use_adaptive": use_adaptive,
        "use_gds": use_gds,
        "use_decomposition": use_decomposition,
        "top_k": top_k,
    }


def api_post(path: str, payload: dict, timeout: int = 600) -> dict:
    """POST JSON to the FastAPI backend (stdlib only); raise RuntimeError on failure."""
    request = urllib.request.Request(
        API_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8")).get("detail", str(e))
        except Exception:
            detail = str(e)
        raise RuntimeError(f"API error {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach GraphRAG API at {API_URL}: {e.reason}")


def api_health() -> bool:
    """True when GET /health answers 200 (backend reachable)."""
    try:
        with urllib.request.urlopen(API_URL + "/health", timeout=10) as response:
            return response.status == 200
    except Exception:
        return False


def set_example_question(question):
    """Callback to set example question in session state."""
    st.session_state.search_query = question


def main():
    """Main application logic."""

    # Initialize components: only the viz-only graph driver lives here now.
    # Retrieval/generation run in FastAPI; gate on its health instead.
    driver = init_graph_driver()

    if not api_health():
        st.error(f"GraphRAG API is unreachable at {API_URL}.")
        st.info("Start it with: uvicorn api.main:app --host 127.0.0.1 --port 8000 "
                "(or set GRAPHRAG_API_URL to its address)")
        st.stop()

    # Sidebar: retrieval settings only (evaluation panel removed from UI;
    # metrics remain available in code/data, just no longer displayed here)
    with st.sidebar:
        st.markdown('<p class="section-title">Retrieval Settings</p>', unsafe_allow_html=True)
        expand_graph, use_adaptive, use_gds, use_decomposition, top_k = render_sidebar_settings()

    # Header with inline status strip
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <p style="color: #5C6265; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem;">GRAPHRAG RESEARCH</p>
            <h1 style="margin-bottom: 0.5rem;">Scientific Literature Q&A</h1>
            <p style="color: #5C6265; font-size: 1.05rem; line-height: 1.5;">Explore scientific literature with evidence-backed answers powered by GraphRAG, hybrid retrieval, and knowledge graphs.</p>
            <div class="status-strip">
                <span class="status-badge"><span class="status-dot"></span>Neo4j</span>
                <span class="status-badge"><span class="status-dot"></span>Vector Search</span>
                <span class="status-badge"><span class="status-dot"></span>Graph Retrieval</span>
                <span class="status-badge"><span class="status-dot"></span>LLM</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Main search interface
    query, system_mode = render_search_bar()

    # Example questions as clickable pills near the search bar (same callbacks)
    st.markdown('<p class="section-title">Try an example</p>', unsafe_allow_html=True)
    pill1, pill2, pill3 = st.columns(3)

    with pill1:
        st.button(
            "Oxidative stress in neurodegeneration",
            on_click=set_example_question,
            args=("What role does oxidative stress play in neurodegenerative diseases?",),
            type="secondary"
        )

    with pill2:
        st.button(
            "Cardiovascular risk factors",
            on_click=set_example_question,
            args=("What are the risk factors associated with cardiovascular disease?",),
            type="secondary"
        )

    with pill3:
        st.button(
            "Mitochondrial dysfunction and cell death",
            on_click=set_example_question,
            args=("How does mitochondrial dysfunction contribute to programmed cell death?",),
            type="secondary"
        )

    # Search button
    if st.button("Search Literature", type="primary"):
        if not query or not query.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Processing your question..."):
                try:
                    # Query flow goes through FastAPI (same branching/payload
                    # the backend implements from the Streamlit settings).
                    payload = build_query_payload(
                        query, system_mode, expand_graph, use_adaptive,
                        use_gds, use_decomposition, top_k,
                    )
                    data = api_post("/query", payload)
                    answer = data["answer"]
                    retrieved_chunks = data["retrieved_chunks"]

                    if not retrieved_chunks:
                        st.warning("No relevant chunks found. Try a different question.")
                        st.stop()

                    # Success status
                    st.markdown(
                        f"""
                        <div style="
                            background: rgba(45, 155, 92, 0.08);
                            border: 1px solid rgba(45, 155, 92, 0.3);
                            border-radius: 8px;
                            padding: 0.75rem 1rem;
                            margin: 1.5rem 0;
                            color: #2D9B5C;
                            font-size: 0.9rem;
                        ">
                            ✓ Retrieved {len(retrieved_chunks)} relevant evidence chunks
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Strategy comes from the API response (computed server-side).
                    strategy = data.get("retrieval_strategy", "n/a")
                    breakdown = retrieved_chunks[0].get("score_breakdown", {}) if retrieved_chunks else {}
                    weights_line = ""
                    if breakdown:
                        weights_line = (
                            f"Fusion α={breakdown.get('alpha', 0):.2f} "
                            f"β={breakdown.get('beta', 0):.2f} "
                            f"γ={breakdown.get('gamma', 0):.2f}"
                        )
                    st.markdown(
                        f"""
                        <div style="
                            background: rgba(15, 82, 87, 0.06);
                            border: 1px solid #E2DFD6;
                            border-radius: 8px;
                            padding: 0.75rem 1rem;
                            margin: 0 0 1.5rem 0;
                            color: #1A1D1E;
                            font-size: 0.9rem;
                        ">
                            <b>Retrieval Strategy:</b> {strategy} &nbsp;·&nbsp; <b>System:</b> {system_mode}
                            {('<br><span style="color: #5C6265;">' + weights_line + '</span>') if weights_line else ''}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Display results in tabs (same data, organized)
                    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

                    answer_tab, evidence_tab, graph_tab = st.tabs(
                        ["Answer", "Evidence", "Knowledge Graph"]
                    )

                    # Answer + evidence + graph come from the API response as-is.
                    with answer_tab:
                        render_answer_card(answer)

                    with evidence_tab:
                        render_evidence_panel(retrieved_chunks)

                    with graph_tab:
                        render_graph_view(driver, data.get("chunk_ids", []))

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Please try again or contact support if the issue persists.")


if __name__ == "__main__":
    main()
