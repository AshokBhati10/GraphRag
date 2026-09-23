"""
Phase 5 - Answer Card Component
Displays the generated answer with styling.
"""

import streamlit as st
import re


def render_answer_card(answer: str):
    """
    Render the generated answer in a styled card.

    Args:
        answer: Generated answer text from the LLM
    """

    st.markdown("### Answer")

    # Highlight citation numbers [1], [2], etc. with coral color
    answer_html = re.sub(
        r'\[(\d+)\]',
        r'<span style="color: #FF6B4A; font-weight: 600;">[\1]</span>',
        answer
    )

    # Render answer in clean card
    st.markdown(
        f"""
        <div style="
            background: #FFFFFF;
            border: 1px solid #E2DFD6;
            border-left: 3px solid #0F5257;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        ">
            <p style="
                font-size: 1rem;
                line-height: 1.7;
                color: #1A1D1E;
                margin: 0;
            ">{answer_html}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
