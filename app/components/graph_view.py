"""
Phase 5 - Graph Visualization Component
Visualizes the knowledge graph around retrieved chunks using PyVis.
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import List
from pyvis.network import Network
import re


def render_graph_view(driver, chunk_ids: List[str], show_similar: bool = True):
    """
    Render an interactive knowledge graph visualization.

    Shows retrieved chunks and their connected entities with MENTIONS
    relationships, plus SEMANTIC_SIMILAR edges between retrieved chunks
    where available.

    Args:
        driver: Neo4j driver connection
        chunk_ids: List of chunk IDs to visualize
        show_similar: Whether to include SEMANTIC_SIMILAR edges (default: True)
    """

    if not chunk_ids:
        st.info("No chunks to visualize")
        return

    st.markdown('<p class="section-title">Knowledge Graph</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #5C6265; font-size: 0.9rem; margin-bottom: 1rem;">Entity relationships across the retrieved literature</p>',
        unsafe_allow_html=True
    )

    try:
        # Query Neo4j for chunks, entities, and relationships
        with driver.session() as session:
            query = """
            MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity)
            WHERE c.chunk_id IN $chunkIds
            RETURN c.chunk_id AS chunk_id,
                   c.text AS chunk_text,
                   e.name AS entity_name,
                   'Entity' AS entity_type
            LIMIT 200
            """

            result = session.run(query, chunkIds=chunk_ids)
            records = list(result)

            if not records:
                st.warning("No graph data found for retrieved chunks")
                return

            # SEMANTIC_SIMILAR edges between retrieved chunks (optional)
            similar_pairs = []
            if show_similar:
                try:
                    sim_result = session.run(
                        """
                        MATCH (a:Chunk)-[r:SEMANTIC_SIMILAR]->(b:Chunk)
                        WHERE a.chunk_id IN $chunkIds AND b.chunk_id IN $chunkIds
                        RETURN DISTINCT a.chunk_id AS from_id, b.chunk_id AS to_id,
                               r.similarity AS similarity
                        LIMIT 200
                        """,
                        chunkIds=chunk_ids,
                    )
                    similar_pairs = list(sim_result)
                except Exception:
                    similar_pairs = []

            # Create PyVis network with light background.
            # Sized to fill the tab width with a tall readable canvas.
            net = Network(
                height="780px",
                width="100%",
                bgcolor="#FFFFFF",
                font_color="#1A1D1E",
                directed=False
            )

            # Configure physics
            net.set_options("""
            {
                "physics": {
                    "enabled": true,
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 100,
                        "springConstant": 0.08
                    },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "timestep": 0.35,
                    "stabilization": {"iterations": 150}
                },
                "nodes": {
                    "font": {
                        "color": "#1A1D1E",
                        "size": 12,
                        "face": "Inter"
                    },
                    "borderWidth": 2
                },
                "edges": {
                    "color": {
                        "color": "#C4C0B5"
                    },
                    "width": 2
                }
            }
            """)

            # Track added nodes to avoid duplicates
            added_chunks = set()
            added_entities = set()

            # Add nodes and edges
            for record in records:
                chunk_id = record["chunk_id"]
                entity_name = record["entity_name"]
                entity_type = record["entity_type"] or "Unknown"

                # Add chunk node (dark teal box, largest)
                if chunk_id not in added_chunks:
                    chunk_text = record["chunk_text"][:100] + "..." if len(record["chunk_text"]) > 100 else record["chunk_text"]
                    net.add_node(
                        chunk_id,
                        label=chunk_id,
                        title=chunk_text,
                        color="#0F5257",
                        size=34,
                        shape="box",
                        font={"color": "#FFFFFF", "size": 12, "face": "Monaco"},
                        borderWidth=0
                    )
                    added_chunks.add(chunk_id)

                # Add entity node (orange/coral dot)
                if entity_name not in added_entities:
                    entity_type = record["entity_type"] if record["entity_type"] else "Entity"
                    net.add_node(
                        entity_name,
                        label=entity_name,
                        title=f"{entity_name} ({entity_type})",
                        color="#FF6B4A",
                        size=20,
                        shape="dot",
                        font={"color": "#1A1D1E", "size": 12, "face": "Inter"},
                        borderWidth=2,
                        borderWidthSelected=3
                    )
                    added_entities.add(entity_name)

                # Add MENTIONS edge (light subtle gray)
                net.add_edge(chunk_id, entity_name, color="#C4C0B5", width=2)

            # Add SEMANTIC_SIMILAR edges between retrieved chunk nodes (green)
            for pair in similar_pairs:
                from_id, to_id = pair["from_id"], pair["to_id"]
                if from_id in added_chunks and to_id in added_chunks:
                    sim = pair["similarity"]
                    title = f"SEMANTIC_SIMILAR ({sim:.3f})" if sim is not None else "SEMANTIC_SIMILAR"
                    net.add_edge(from_id, to_id, color="#2D9B5C", width=3,
                                 dashes=True, title=title)

            # Generate HTML directly as a string (no temp file: Windows
            # file locking (WinError 32) made the NamedTemporaryFile
            # round-trip unreliable). Same PyVis output, kept in memory.
            html_content = net.generate_html()

            # Style the emitted #mynetwork CSS rule (pyvis 0.3.2 emits the
            # canvas size as a stylesheet rule, not inline div styles).
            # Rounded corners/overflow only — size comes from Network().
            html_content = html_content.replace(
                "#mynetwork {",
                "#mynetwork {\n border-radius: 8px;\n overflow: hidden;",
                1,
            )

            # Also ensure body and html have proper sizing
            html_content = re.sub(
                r'<body>',
                '<body style="margin: 0; padding: 0; overflow: hidden; height: 780px;">',
                html_content
            )

            html_content = re.sub(
                r'<html>',
                '<html style="height: 780px;">',
                html_content
            )

            # Display graph using st.components.v1.html() for interactive PyVis content
            # Height matches the 780px canvas; width fills the tab column.
            components.html(html_content, height=780, scrolling=False)

            # Legend
            st.markdown(
                """
                <div style="
                    margin-top: 1rem;
                    padding: 1rem;
                    background: #F3F1EA;
                    border: 1px solid #E2DFD6;
                    border-radius: 8px;
                ">
                    <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.85rem;">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="display: inline-block; width: 16px; height: 16px; background: #0F5257; border-radius: 2px;"></span>
                            <span style="color: #1A1D1E;">Evidence Chunk</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="display: inline-block; width: 16px; height: 16px; background: #FF6B4A; border-radius: 50%;"></span>
                            <span style="color: #1A1D1E;">Entity</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="display: inline-block; width: 20px; height: 2px; background: #C4C0B5;"></span>
                            <span style="color: #1A1D1E;">MENTIONS relationship</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="display: inline-block; width: 20px; height: 2px; background: #2D9B5C;"></span>
                            <span style="color: #1A1D1E;">SEMANTIC_SIMILAR relationship</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Failed to generate graph visualization: {str(e)}")
        st.info("Graph visualization is optional. The answer and evidence are still available above.")
