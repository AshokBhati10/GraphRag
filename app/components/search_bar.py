"""
Phase 5 - Search Bar Component
Provides the main search interface with controls for GraphRAG parameters.
"""

import streamlit as st


def render_search_bar():
    """
    Render the search bar and control panel.

    Returns:
        Tuple of (query, expand_graph, use_gds, top_k, use_decomposition,
        system_mode, use_adaptive)
        - system_mode: "Modified GraphRAG" or "Baseline"
        - use_adaptive: whether adaptive retrieval may skip graph expansion
    """

    # Search input
    query = st.text_input(
        "Question",
        placeholder="Ask a question about the scientific literature...",
        key="search_query",
        label_visibility="collapsed"
    )

    st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)

    # System mode selection (Baseline vs Modified GraphRAG)
    system_mode = st.radio(
        "System",
        options=["Modified GraphRAG", "Baseline"],
        index=0,
        horizontal=True,
        help="Baseline: required expansion + legacy fusion. Modified: adaptive retrieval, graph-aware fusion, PageRank."
    )

    st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)

    # Retrieval configuration heading
    st.markdown(
        """
        <p style="
            color: #5C6265;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 0 0 0.75rem 0;
        ">Retrieval Configuration</p>
        """,
        unsafe_allow_html=True
    )

    # Controls in columns
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        expand_graph = st.checkbox(
            "Graph Expansion",
            value=True,
            help="Expand retrieval via entity relationships"
        )

    with col2:
        use_adaptive = st.checkbox(
            "Adaptive",
            value=True,
            help="Let the system skip expansion when vector hits are confident"
        )

    with col3:
        use_gds = st.checkbox(
            "PageRank Reranking",
            value=True,
            help="Use PageRank to rerank results"
        )

    with col4:
        use_decomposition = st.checkbox(
            "Query Decomposition",
            value=False,
            help="Break complex queries into sub-queries"
        )

    with col5:
        top_k = st.slider(
            "Top K Results",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of chunks to retrieve"
        )

    st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)

    return query, expand_graph, use_gds, top_k, use_decomposition, system_mode, use_adaptive
