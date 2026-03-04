"""
Enhanced KG Viewer — Interactive graph with zoom, edge triplet tooltips, and stats.
Called from app.py's tab5.
"""
import streamlit as st
import pandas as pd
import html as html_mod


def render_kg_viewer(company, load_kg_graph_fn, clean_display_text_fn,
                     format_risk_name_fn, format_capability_fn):
    """Render the enhanced KG Viewer in the current Streamlit context."""

    st.markdown(f"### 🕸️ Knowledge Graph — {company}")
    st.caption("Scroll to zoom · Drag to pan · Hover edges for full triplet · "
               "Use navigation buttons (bottom-left) · Press fullscreen ⛶ to expand")

    G = load_kg_graph_fn()

    # Find company node using multiple naming patterns (TTL + CSV)
    company_node = None
    candidates = [f"Company_{company}", f"Company:{company}", company.upper(), company]
    for c in candidates:
        if c in G:
            if company_node is None or G.out_degree(c) > G.out_degree(company_node):
                company_node = c

    if company_node is None:
        st.warning(f"No KG data found for {company}. Run `python kg_builder.py` first.")
        return

    # Also find secondary company nodes to merge subgraphs
    all_company_nodes = [c for c in candidates if c in G]

    # Build company-specific subgraph (2-hop from ALL company nodes)
    sub_nodes = set(all_company_nodes)
    for cn in all_company_nodes:
        for n1 in G.neighbors(cn):
            sub_nodes.add(n1)
            for n2 in G.neighbors(n1):
                sub_nodes.add(n2)

    subgraph = G.subgraph(sub_nodes)

    # All circles, different colors only
    node_colors = {
        "Company":            "#2962a8",
        "Mission":            "#e74c3c",
        "StrategicObjective": "#27ae60",
        "Capability":         "#f39c12",
        "Initiative":         "#8e44ad",
        "RiskTheme":          "#e67e22",
        "FinancialMetric":    "#3498db",
        "HumanCapitalMetric": "#1abc9c",
    }

    node_sizes = {
        "Company": 35,
        "Mission": 28,
    }
    DEFAULT_SIZE = 22

    # Edge color map
    edge_colors = {
        "hasMission":       "#e74c3c",
        "statesObjective":  "#27ae60",
        "buildsCapability": "#f39c12",
        "pursues":          "#8e44ad",
        "exposedTo":        "#e67e22",
        "supportsObjective":"#2ecc71",
        "initiativeEnables":"#9b59b6",
    }

    # ── Stats counts ──
    type_counts = {}
    for node in subgraph.nodes():
        lt = G.nodes[node].get(":LABEL", "Other")
        type_counts[lt] = type_counts.get(lt, 0) + 1

    rel_counts = {}
    for _, _, data in subgraph.edges(data=True):
        rel = data.get("relation", "unknown")
        rel_counts[rel] = rel_counts.get(rel, 0) + 1

    # ── Build pyvis graph ──
    try:
        from pyvis.network import Network
        import tempfile

        net = Network(
            height="700px", width="100%",
            bgcolor="#fafbfd", font_color="#333333", directed=True,
        )
        net.toggle_physics(True)
        net.set_options("""{
            "nodes": {
                "shape": "dot",
                "font": {"size": 13, "face": "Segoe UI, sans-serif",
                         "strokeWidth": 3, "strokeColor": "#ffffff"},
                "borderWidth": 2,
                "borderWidthSelected": 4,
                "shadow": {"enabled": true, "size": 5, "x": 2, "y": 2}
            },
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.9}},
                "font": {"size": 10, "align": "middle",
                         "strokeWidth": 2, "strokeColor": "#ffffff"},
                "smooth": {"type": "curvedCW", "roundness": 0.15},
                "width": 2, "hoverWidth": 3, "selectionWidth": 3
            },
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -4000,
                    "springLength": 180,
                    "springConstant": 0.04,
                    "damping": 0.12
                },
                "stabilization": {"iterations": 200}
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "zoomView": true,
                "dragView": true,
                "navigationButtons": true,
                "keyboard": {"enabled": true}
            }
        }""")

        # ── Add nodes ──
        for node in subgraph.nodes():
            node_data = G.nodes[node]
            label_type = node_data.get(":LABEL", "Unknown")
            full_text = node_data.get("text", "") or node_data.get("name", "") or ""
            display_name = full_text if full_text else node.split(":")[-1]

            # Short label for graph
            label_short = clean_display_text_fn(display_name) if display_name else node.split(":")[-1]
            if label_type == "RiskTheme":
                label_short = format_risk_name_fn(display_name)
            elif label_type == "Capability":
                label_short = format_capability_fn(display_name)
            if len(label_short) > 45:
                label_short = label_short[:42] + "..."

            # Full clean text
            full_clean = clean_display_text_fn(full_text) if full_text else ""
            if label_type == "RiskTheme":
                full_clean = format_risk_name_fn(full_text)

            # Plain-text tooltip (no HTML)
            conn_lines = []
            for pred in G.predecessors(node):
                r = G[pred][node].get("relation", "?")
                pred_name = G.nodes[pred].get("name", "") or G.nodes[pred].get("text", "") or pred
                if G.nodes[pred].get(":LABEL", "") == "RiskTheme":
                    pred_name = format_risk_name_fn(pred_name)
                conn_lines.append(f"  IN:  [{G.nodes[pred].get(':LABEL','')}] --{r}--> this")
            for succ in G.successors(node):
                r = G[node][succ].get("relation", "?")
                succ_name = G.nodes[succ].get("name", "") or G.nodes[succ].get("text", "") or succ
                if G.nodes[succ].get(":LABEL", "") == "RiskTheme":
                    succ_name = format_risk_name_fn(succ_name)
                conn_lines.append(f"  OUT: this --{r}--> [{G.nodes[succ].get(':LABEL','')}]")

            tooltip_parts = [
                f"[{label_type}]",
                f"Text: {full_clean[:250]}",
                "",
                "Connections:",
            ]
            tooltip_parts.extend(conn_lines[:8])
            tooltip = "\n".join(tooltip_parts)

            color = node_colors.get(label_type, "#95a5a6")
            size = node_sizes.get(label_type, DEFAULT_SIZE)

            net.add_node(
                node,
                label=label_short,
                color=color,
                size=size,
                shape="dot",
                title=tooltip,
                font={"size": 13 if label_type == "Company" else 11},
            )

        # ── Add edges with plain-text triplet tooltips ──
        for u, v, data in subgraph.edges(data=True):
            rel = data.get("relation", "")
            u_label = G.nodes[u].get(":LABEL", "")
            v_label = G.nodes[v].get(":LABEL", "")

            u_text = clean_display_text_fn(
                G.nodes[u].get("text", "") or G.nodes[u].get("name", "") or u
            )
            v_text = clean_display_text_fn(
                G.nodes[v].get("text", "") or G.nodes[v].get("name", "") or v
            )
            if u_label == "RiskTheme":
                u_text = format_risk_name_fn(
                    G.nodes[u].get("name", "") or G.nodes[u].get("text", "") or u
                )
            if v_label == "RiskTheme":
                v_text = format_risk_name_fn(
                    G.nodes[v].get("name", "") or G.nodes[v].get("text", "") or v
                )

            # Plain text triplet — no HTML tags
            edge_tooltip = (
                f"TRIPLE\n"
                f"Subject:   [{u_label}] {u_text[:120]}\n"
                f"Predicate: {rel}\n"
                f"Object:    [{v_label}] {v_text[:120]}"
            )

            ec = edge_colors.get(rel, "#888888")
            net.add_edge(
                u, v,
                label=rel,
                title=edge_tooltip,
                color={"color": ec, "highlight": "#1a1a2e", "hover": ec},
                width=2,
            )

        # ── Save, inject legend + fullscreen, and render ──
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".html", mode="w", encoding="utf-8"
        ) as f:
            net.save_graph(f.name)
            with open(f.name, "r", encoding="utf-8") as fr:
                graph_html = fr.read()

            # Build legend HTML — only show types that exist in this subgraph
            legend_items = ""
            for lbl, cnt in sorted(type_counts.items()):
                clr = node_colors.get(lbl, "#95a5a6")
                legend_items += (
                    f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'border-radius:50%;background:{clr};flex-shrink:0;"></span>'
                    f'<span style="font-size:12px;color:#333;">{lbl} ({cnt})</span>'
                    f'</div>'
                )

            # Legend overlay + fullscreen button + tooltip styling
            injected = """
            <style>
                div.vis-tooltip {
                    background-color: #ffffff !important;
                    border: 1px solid #ccc !important;
                    border-radius: 6px !important;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.12) !important;
                    padding: 8px 12px !important;
                    max-width: 450px !important;
                    font-family: 'Segoe UI', sans-serif !important;
                    font-size: 12px !important;
                    white-space: pre-line !important;
                    line-height: 1.5 !important;
                    color: #333 !important;
                }
                div.vis-network div.vis-navigation div.vis-button {
                    background-color: rgba(41,98,168,0.12) !important;
                    border: 1px solid #2962a8 !important;
                    border-radius: 6px !important;
                }
                div.vis-network div.vis-navigation div.vis-button:hover {
                    background-color: rgba(41,98,168,0.3) !important;
                }
                #kg-legend {
                    position: absolute; top: 12px; right: 12px; z-index: 999;
                    background: rgba(255,255,255,0.92); backdrop-filter: blur(6px);
                    border: 1px solid #ddd; border-radius: 8px;
                    padding: 10px 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }
                #kg-legend h4 { margin: 0 0 6px 0; font-size: 13px; color: #1a3a5c; }
                #fullscreen-btn {
                    position: absolute; top: 12px; left: 12px; z-index: 999;
                    background: rgba(41,98,168,0.9); color: white;
                    border: none; border-radius: 6px; padding: 6px 14px;
                    font-size: 13px; cursor: pointer; font-weight: 600;
                }
                #fullscreen-btn:hover { background: rgba(41,98,168,1); }
            </style>
            <div id="kg-legend">
                <h4>Legend</h4>
                """ + legend_items + """
                <div style="margin-top:6px;padding-top:6px;border-top:1px solid #eee;font-size:11px;color:#888;">
                    Nodes: """ + str(subgraph.number_of_nodes()) + """ · Edges: """ + str(subgraph.number_of_edges()) + """
                </div>
            </div>
            <button id="fullscreen-btn" onclick="goFullscreen()">⛶ Fullscreen</button>
            <script>
            function goFullscreen() {
                var el = document.documentElement;
                if (el.requestFullscreen) el.requestFullscreen();
                else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
                else if (el.msRequestFullscreen) el.msRequestFullscreen();
            }
            </script>
            """
            graph_html = graph_html.replace("</head>",
                "<style>body{position:relative;}</style></head>")
            graph_html = graph_html.replace("</body>", injected + "</body>")

        st.components.v1.html(graph_html, height=750, scrolling=False)

    except ImportError:
        st.warning("pyvis not installed. Install with `pip install pyvis`")
        st.markdown("#### Graph Structure (text view)")
        for u, v, data in subgraph.edges(data=True):
            u_label = G.nodes[u].get(":LABEL", "")
            v_label = G.nodes[v].get(":LABEL", "")
            rel = data.get("relation", "")
            u_text = clean_display_text_fn(
                G.nodes[u].get("text", "") or G.nodes[u].get("name", "") or ""
            )
            v_text = clean_display_text_fn(
                G.nodes[v].get("text", "") or G.nodes[v].get("name", "") or ""
            )
            st.markdown(
                f"- **[{u_label}]** {u_text[:60]} →`{rel}`→ "
                f"**[{v_label}]** {v_text[:60]}"
            )

    # ── Full Triplet Table ──
    with st.expander("📋 Full Triplet Table (click to expand)"):
        triplet_rows = []
        for u, v, data in subgraph.edges(data=True):
            u_label = G.nodes[u].get(":LABEL", "")
            v_label = G.nodes[v].get(":LABEL", "")
            rel = data.get("relation", "")
            u_text = G.nodes[u].get("text", "") or G.nodes[u].get("name", "") or u
            v_text = G.nodes[v].get("text", "") or G.nodes[v].get("name", "") or v
            if u_label == "RiskTheme":
                u_text = format_risk_name_fn(u_text)
            if v_label == "RiskTheme":
                v_text = format_risk_name_fn(v_text)
            if u_label == "Capability":
                u_text = format_capability_fn(u_text)
            if v_label == "Capability":
                v_text = format_capability_fn(v_text)
            triplet_rows.append({
                "Subject Type": u_label,
                "Subject": clean_display_text_fn(u_text)[:80],
                "Predicate": rel,
                "Object Type": v_label,
                "Object": clean_display_text_fn(v_text)[:80],
            })
        if triplet_rows:
            st.dataframe(
                pd.DataFrame(triplet_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No edges found.")
