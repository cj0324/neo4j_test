import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components

from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama

from graph_app import build_langgraph_app


st.set_page_config(page_title="Neo4j GraphRAG (LangGraph)", layout="wide")


def connect_graph(url: str, username: str, password: str) -> Neo4jGraph:
    g = Neo4jGraph(url=url, username=username, password=password)
    g.refresh_schema()
    return g


def pick_display_from_props(props: dict, node_type: str | None = None) -> str | None:
    """
    일반화 라벨 선택 규칙:
    1) 대표 문자열 키 우선 (Name/title/displayName 등)
    2) 숫자형 프로퍼티 우선 (Euro/amount/score 등 우선키 -> 없으면 아무 숫자)
    3) 마지막 fallback: props 첫 번째 값 (순서 완전 보장은 아님)
    """
    if not props:
        return None

    preferred_text_keys = [
        "Name", "name", "title", "Title",
        "displayName", "DisplayName",
        "key", "code", "id"
    ]
    for k in preferred_text_keys:
        v = props.get(k)
        if v is not None and str(v).strip() != "":
            return str(v)

    preferred_numeric_keys = [
        "Euro", "euro", "amount", "Amount",
        "value", "Value", "score", "Score",
        "price", "Price", "count", "Count",
        "total", "Total", "salary", "Salary"
    ]
    for k in preferred_numeric_keys:
        v = props.get(k)
        if isinstance(v, (int, float)) and v is not None:
            return f"{k}: {v}"

    for k, v in props.items():
        if isinstance(v, (int, float)) and v is not None:
            return f"{k}: {v}"

    first_key = next(iter(props.keys()), None)
    if first_key is not None and props[first_key] is not None:
        return f"{first_key}: {props[first_key]}"

    return None


def render_pyvis(subgraph: dict, height_px: int = 750):
    net = Network(height=f"{height_px}px", width="100%", directed=False)
    net.force_atlas_2based()

    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])

    for n in nodes:
        props = n.get("props", {}) or {}
        node_type = n.get("type")

        display = pick_display_from_props(props, node_type=node_type)
        if not display:
            display = n.get("name") or node_type or n.get("id")

        tooltip_lines = []
        if node_type:
            tooltip_lines.append(f"type: {node_type}")
        for k, v in list(props.items())[:12]:
            tooltip_lines.append(f"{k}: {v}")
        title = "<br>".join(tooltip_lines)

        net.add_node(n["id"], label=str(display), title=title)

    for e in edges:
        net.add_edge(e["source"], e["target"], label=e.get("label", ""))

    html = net.generate_html()
    components.html(html, height=height_px, scrolling=True)


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Connection")
    neo4j_url = st.text_input("NEO4J_URL", value="bolt://localhost:7687")
    neo4j_user = st.text_input("NEO4J_USER", value="neo4j")
    neo4j_pass = st.text_input("NEO4J_PASS", value="/", type="password")

    st.header("LLM (Ollama)")
    model = st.text_input("MODEL", value="qwen3:8b")
    temperature = st.slider("temperature", 0.0, 1.0, 0.0, 0.1)

    st.header("Graph Expansion")
    two_hop = st.toggle("2-hop 확장", value=False)  # ✅ 토글 추가

    do_connect = st.button("Connect / Refresh")


if do_connect or "graph" not in st.session_state:
    try:
        graph = connect_graph(neo4j_url, neo4j_user, neo4j_pass)
        llm = ChatOllama(model=model, temperature=temperature)
        invoke = build_langgraph_app(graph, llm)

        st.session_state["graph"] = graph
        st.session_state["schema"] = graph.schema
        st.session_state["invoke"] = invoke
        st.session_state["current_subgraph"] = None

        st.success("Connected and schema refreshed!")
    except Exception as e:
        st.error(f"Connection failed: {e}")


graph: Neo4jGraph | None = st.session_state.get("graph")
schema: str = st.session_state.get("schema", "")
invoke = st.session_state.get("invoke")


# ----------------------------
# Main UI
# ----------------------------
st.title("Neo4j + LangChain + LangGraph + Streamlit")
st.caption("검색하면 생성된 Cypher/결과/서브그래프를 보여주고, 2-hop 확장도 지원합니다.")

# ✅ 좌/우 메인 컬럼
left_col, right_col = st.columns([1, 2])

# =========================
# LEFT COLUMN
# =========================
with left_col:
    # -------- Graph Overview (위)
    st.subheader("Graph Overview")

    if graph:
        try:
            labels = graph.query("CALL db.labels() YIELD label RETURN label")
            rels = graph.query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")

            st.write(f"- Labels: {len(labels)}")
            st.write(f"- Relationship Types: {len(rels)}")
        except Exception as e:
            st.warning(f"Overview query failed: {e}")

        if st.button("Load Sample Subgraph (200 edges)"):
            try:
                sg = graph.query("""
                MATCH (n)-[r]-(m)
                RETURN elementId(n) AS src_id, labels(n) AS src_labels, properties(n) AS src_props,
                       elementId(m) AS dst_id, labels(m) AS dst_labels, properties(m) AS dst_props,
                       type(r) AS rel_type
                LIMIT 200
                """)

                nodes = {}
                edges = []
                for row in sg:
                    s = row.get("src_id")
                    t = row.get("dst_id")
                    if s and s not in nodes:
                        nodes[s] = {
                            "id": s,
                            "type": ":".join(row.get("src_labels") or []),
                            "props": row.get("src_props") or {}
                        }
                    if t and t not in nodes:
                        nodes[t] = {
                            "id": t,
                            "type": ":".join(row.get("dst_labels") or []),
                            "props": row.get("dst_props") or {}
                        }
                    if s and t and row.get("rel_type"):
                        edges.append({"source": s, "target": t, "label": row["rel_type"]})

                st.session_state["current_subgraph"] = {
                    "nodes": list(nodes.values()),
                    "edges": edges
                }
                st.success("Sample subgraph loaded.")
            except Exception as e:
                st.error(f"Failed to load sample subgraph: {e}")
    else:
        st.info("사이드바에서 Connect / Refresh를 눌러 연결하세요.")

    st.divider()

    # -------- Ask a Question (아래)
    st.subheader("Ask a Question")

    question = st.text_input(
        "질문을 입력하세요",
        value="Barcelona 팀의 최고 연봉 선수는?"
    )

    run = st.button("Search")

    if run:
        if not (graph and invoke and schema):
            st.error("먼저 Connect / Refresh로 Neo4j/LLM 연결을 해주세요.")
        else:
            hop = 2 if two_hop else 1
            out = invoke(question, schema, hop)

            if out.get("error"):
                st.error(out["error"])
                st.text_area(
                    "Generated Cypher (debug)",
                    value=out.get("cypher", ""),
                    height=160
                )
            else:
                st.text_area(
                    "Generated Cypher",
                    value=out.get("cypher", ""),
                    height=160
                )
                st.write("Answer")
                st.write(out.get("answer", ""))

                st.session_state["current_subgraph"] = out.get(
                    "subgraph",
                    {"nodes": [], "edges": []}
                )

# =========================
# RIGHT COLUMN
# =========================
with right_col:
    st.subheader("Graph Visualization")

    subgraph = st.session_state.get("current_subgraph")
    if subgraph:
        render_pyvis(subgraph)
    else:
        st.info("샘플 그래프를 로드하거나 질문(Search)을 실행하면 그래프가 표시됩니다.")

# st.title("Neo4j + LangChain + LangGraph + Streamlit")
# st.caption("2-hop 토글로 확장 범위를 바꿀 수 있고, 노드 라벨은 props 기반으로 자동 선택합니다.")

# left, right = st.columns([1, 1])

# with left:
#     st.subheader("Graph Overview")

#     if graph:
#         try:
#             labels = graph.query("CALL db.labels() YIELD label RETURN label")
#             rels = graph.query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")

#             st.write(f"- Labels: {len(labels)}")
#             st.write(f"- Relationship Types: {len(rels)}")
#         except Exception as e:
#             st.warning(f"Overview query failed: {e}")

#         if st.button("Load Sample Subgraph (200 edges)"):
#             try:
#                 sg = graph.query("""
#                 MATCH (n)-[r]-(m)
#                 RETURN elementId(n) AS src_id, labels(n) AS src_labels, properties(n) AS src_props,
#                        elementId(m) AS dst_id, labels(m) AS dst_labels, properties(m) AS dst_props,
#                        type(r) AS rel_type
#                 LIMIT 200
#                 """)

#                 nodes = {}
#                 edges = []
#                 for row in sg:
#                     s = row.get("src_id")
#                     t = row.get("dst_id")
#                     if s and s not in nodes:
#                         nodes[s] = {"id": s, "type": ":".join(row.get("src_labels") or []), "props": row.get("src_props") or {}}
#                     if t and t not in nodes:
#                         nodes[t] = {"id": t, "type": ":".join(row.get("dst_labels") or []), "props": row.get("dst_props") or {}}
#                     if s and t and row.get("rel_type"):
#                         edges.append({"source": s, "target": t, "label": row["rel_type"]})

#                 st.session_state["current_subgraph"] = {"nodes": list(nodes.values()), "edges": edges}
#                 st.success("Sample subgraph loaded.")
#             except Exception as e:
#                 st.error(f"Failed to load sample subgraph: {e}")
#     else:
#         st.info("사이드바에서 Connect / Refresh를 눌러 연결하세요.")


# with right:
#     st.subheader("Ask a question")

#     question = st.text_input("질문을 입력하세요", value="Barcelona 팀의 최고 연봉 선수는?")
#     run = st.button("Search")

#     if run:
#         if not (graph and invoke and schema):
#             st.error("먼저 Connect / Refresh로 Neo4j/LLM 연결을 해주세요.")
#         else:
#             hop = 2 if two_hop else 1
#             out = invoke(question, schema, hop)

#             if out.get("error"):
#                 st.error(out["error"])
#                 st.text_area("Generated Cypher (debug)", value=out.get("cypher", ""), height=160)
#             else:
#                 st.text_area("Generated Cypher", value=out.get("cypher", ""), height=160)
#                 st.write("Result (raw)")
#                 st.json(out.get("result", []))
#                 st.write("Answer (summary)")
#                 st.write(out.get("answer", ""))

#                 st.session_state["current_subgraph"] = out.get("subgraph", {"nodes": [], "edges": []})


# st.divider()
# st.subheader("Graph Visualization")

# subgraph = st.session_state.get("current_subgraph")
# if subgraph:
#     render_pyvis(subgraph)
# else:
#     st.info("샘플 그래프를 로드하거나 질문(Search)을 실행하면 그래프가 표시됩니다.")
