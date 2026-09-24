"""
Phase 5 - Evidence Panel Component
Displays retrieved chunks as evidence with scores.
"""

import streamlit as st
from typing import List, Dict


def render_evidence_panel(chunks: List[Dict]):
    """
    Render the evidence panel showing retrieved chunks with scores.

    Args:
        chunks: List of retrieved chunks with scores
    """

    st.markdown("### Evidence")
    st.markdown(
        '<p style="color: #5C6265; font-size: 0.9rem; margin-bottom: 1rem;">Retrieved evidence used to generate this answer</p>',
        unsafe_allow_html=True
    )

    with st.expander(f"Evidence ({len(chunks)} chunks)", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            text = chunk.get("text", "")
            vector_score = chunk.get("vector_score", 0.0)
            shared_entities = chunk.get("shared_entities", 0)
            depth = chunk.get("depth", 0)
            combined_score = chunk.get("combined_score", 0.0)
            pagerank_score = chunk.get("pagerank_score", 0.0)
            final_score = chunk.get("final_score", combined_score)
            breakdown = chunk.get("score_breakdown", {}) or {}
            proximity = breakdown.get("proximity_score", 1.0 / (1 + max(int(depth), 0)))
            explanation = chunk.get("score_explanation", "")
            strategy = chunk.get("retrieval_strategy", "")

            # Collapsed per-chunk cards (same fields, details on demand)
            with st.expander(f"[{i}] {chunk_id} — final {final_score:.3f}", expanded=(i == 1)):
                # Truncate text for display
                display_text = text[:300] + "..." if len(text) > 300 else text

                why_html = ""
                if explanation:
                    why_html = (
                        '<p style="color: #5C6265; font-size: 0.78rem; '
                        'margin-top: 0.75rem; font-family: Monaco, monospace;">'
                        f"Why selected: {explanation}</p>"
                    )
                strategy_html = ""
                if strategy:
                    strategy_html = (
                        '<p style="color: #5C6265; font-size: 0.78rem; '
                        f'margin-top: 0.25rem;">Strategy: {strategy}</p>'
                    )

                st.markdown(
                f"""
                <div style="
                    background: #FFFFFF;
                    border: 1px solid #E2DFD6;
                    border-left: 3px solid #FF6B4A;
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                ">
                    <p style="
                        font-family: 'Monaco', 'Menlo', monospace;
                        font-size: 0.85rem;
                        color: #FF6B4A;
                        font-weight: 600;
                        margin-bottom: 0.75rem;
                    ">
                        [{i}] {chunk_id}
                    </p>
                    <p style="
                        color: #1A1D1E;
                        line-height: 1.6;
                        margin-bottom: 1rem;
                        font-size: 0.9rem;
                    ">
                        {display_text}
                    </p>
                    <div style="
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
                        gap: 0.75rem;
                        padding-top: 0.75rem;
                        border-top: 1px solid #E2DFD6;
                    ">
                        <div>
                            <div style="font-size: 0.75rem; color: #5C6265; margin-bottom: 0.25rem;">Vector</div>
                            <div style="font-family: 'Monaco', monospace; font-size: 0.85rem; color: #1A1D1E; font-weight: 600;">{vector_score:.3f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #5C6265; margin-bottom: 0.25rem;">Entities</div>
                            <div style="font-family: 'Monaco', monospace; font-size: 0.85rem; color: #1A1D1E; font-weight: 600;">{shared_entities}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #5C6265; margin-bottom: 0.25rem;">Graph depth</div>
                            <div style="font-family: 'Monaco', monospace; font-size: 0.85rem; color: #1A1D1E; font-weight: 600;">{depth} ({proximity:.2f})</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #5C6265; margin-bottom: 0.25rem;">Combined</div>
                            <div style="font-family: 'Monaco', monospace; font-size: 0.85rem; color: #1A1D1E; font-weight: 600;">{combined_score:.3f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #5C6265; margin-bottom: 0.25rem;">PageRank</div>
                            <div style="font-family: 'Monaco', monospace; font-size: 0.85rem; color: #1A1D1E; font-weight: 600;">{pagerank_score:.3f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #5C6265; margin-bottom: 0.25rem;">Final</div>
                            <div style="font-family: 'Monaco', monospace; font-size: 0.85rem; color: #0F5257; font-weight: 600;">{final_score:.3f}</div>
                        </div>
                    </div>
                    {why_html}
                    {strategy_html}
                </div>
                """,
                    unsafe_allow_html=True
                )
