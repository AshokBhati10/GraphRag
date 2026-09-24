"""
Phase 5 - Interactive Streamlit Demo for GraphRAG Scientific Literature
Full-featured Q&A interface with query decomposition, PageRank reranking, and graph visualization.
"""

import streamlit as st
import os
import sys

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="GraphRAG Scientific Literature Q&A",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from neo4j import GraphDatabase

from src.config import settings
from src.embeddings import get_embedder, get_retrieval_model_name
from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import build_rag_prompt
from src.retrieval.decompose import retrieve_decomposed
from src.retrieval.retrieve import retrieve
from src.retrieval.retrieve_with_pagerank import retrieve_with_pagerank

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
def init_neo4j_driver():
    """Initialize Neo4j driver (cached)."""
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
        st.info("Please ensure Neo4j is running and credentials are correct in .env file")
        return None


@st.cache_resource
def init_embedding_model():
    """Initialize sentence transformer model (cached).

    Uses the central embedding provider so the model always matches
    settings.EMBEDDING_MODEL_NAME (PDF baseline: all-MiniLM-L6-v2).
    """
    try:
        return get_embedder(get_retrieval_model_name())
    except Exception as e:
        st.error(f"Failed to load embedding model: {str(e)}")
        return None


@st.cache_resource
def init_llm_client():
    """Initialize LLM client (cached)."""
    try:
        return LLMClient(
            backend=settings.LLM_BACKEND,
            model=settings.LLM_MODEL,
            temperature=0.0
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM client: {str(e)}")
        return None


def set_example_question(question):
    """Callback to set example question in session state."""
    st.session_state.search_query = question


def main():
    """Main application logic."""

    # Initialize components
    driver = init_neo4j_driver()
    embedding_model = init_embedding_model()
    llm_client = init_llm_client()

    if not driver or not embedding_model or not llm_client:
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
            <p style="color: #5C6265; font-size: 1.05rem; line-height: 1.5;">Evidence-backed answers from scientific literature, powered by hybrid retrieval and knowledge graphs.</p>
            <div class="status-strip">
                <span class="status-badge"><span class="status-dot"></span>Neo4j</span>
                <span class="status-badge"><span class="status-dot"></span>Embedding</span>
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
                    is_baseline = (system_mode == "Baseline")
                    # Step 1: Retrieval
                    if use_decomposition:
                        st.info("Decomposing query into sub-questions...")
                        retrieved_chunks = retrieve_decomposed(
                            driver=driver,
                            embedding_model=embedding_model,
                            question=query,
                            llm_client=llm_client,
                            top_k_per_subq=3,
                            expand_graph=expand_graph
                        )
                        # Limit to top_k after merging
                        retrieved_chunks = retrieved_chunks[:top_k]
                    elif is_baseline:
                        # PDF baseline: required expansion + legacy fusion,
                        # no PageRank, no adaptive skipping.
                        retrieved_chunks = retrieve(
                            driver=driver,
                            embedding_model=embedding_model,
                            query=query,
                            top_k=top_k,
                            expand_graph=expand_graph,
                            adaptive_enabled=False,
                            alpha=0.7,
                            beta=0.3,
                            gamma=0.0,
                        )
                        for c in retrieved_chunks:
                            c["pagerank_score"] = 0.0
                            c["final_score"] = c["combined_score"]
                    else:
                        retrieved_chunks = retrieve_with_pagerank(
                            driver=driver,
                            embedding_model=embedding_model,
                            query=query,
                            top_k=top_k,
                            expand_graph=expand_graph,
                            use_pagerank=use_gds,
                            adaptive_enabled=use_adaptive
                        )

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

                    # Retrieval strategy summary (adaptive decision + fusion weights)
                    strategy = retrieved_chunks[0].get("retrieval_strategy", "n/a") if retrieved_chunks else "n/a"
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

                    # Step 2: Generate answer
                    with st.spinner("Generating answer..."):
                        prompt = build_rag_prompt(query, retrieved_chunks)
                        answer = llm_client.generate(prompt)

                    # Display results in tabs (same data, organized)
                    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

                    answer_tab, evidence_tab, graph_tab = st.tabs(
                        ["Answer", "Evidence", "Knowledge Graph"]
                    )

                    with answer_tab:
                        render_answer_card(answer)

                    with evidence_tab:
                        render_evidence_panel(retrieved_chunks)

                    with graph_tab:
                        chunk_ids = [c["chunk_id"] for c in retrieved_chunks]
                        render_graph_view(driver, chunk_ids)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Please try again or contact support if the issue persists.")


if __name__ == "__main__":
    main()
