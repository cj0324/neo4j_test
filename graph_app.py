import re
from typing import TypedDict, Any, Dict, List

from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate


class GraphState(TypedDict, total=False):
    question: str
    schema: str
    cypher: str
    result: Any
    answer: str
    subgraph: Dict[str, Any]
    error: str
    hop: int  # ✅ 1 or 2


# ----------------------------
# Cypher safety
# ----------------------------
READ_ONLY_PREFIX = re.compile(r"^\s*(MATCH|WITH|RETURN|UNWIND|CALL)\b", re.IGNORECASE)
FORBIDDEN = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|DROP|REMOVE|LOAD\s+CSV)\b", re.IGNORECASE)

def is_safe_readonly_cypher(cypher: str) -> bool:
    if not cypher or not READ_ONLY_PREFIX.search(cypher):
        return False
    if FORBIDDEN.search(cypher):
        return False
    return True


# ----------------------------
# elementId usage normalization
# - Fixes: p.elementId -> elementId(p)
# ----------------------------
DOT_ELEMENTID = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.elementId\b")

def normalize_elementid_usage(cypher: str) -> str:
    return DOT_ELEMENTID.sub(r"elementId(\1)", cypher)


# ----------------------------
# Extractors from result
# ----------------------------
EID_KEYS = ["eid", "elementId", "nid", "mid", "src_id", "dst_id"]
NAME_KEYS = ["Name", "name", "title", "Title", "displayName", "DisplayName"]

def extract_eids_from_result(result: Any) -> List[str]:
    candidates: List[str] = []
    if not isinstance(result, list):
        return candidates

    for row in result:
        if not isinstance(row, dict):
            continue

        for k in EID_KEYS:
            if k in row and row[k] is not None:
                candidates.append(str(row[k]))

        for _, v in row.items():
            if isinstance(v, dict):
                for k in EID_KEYS + ["id"]:
                    if k in v and v[k] is not None:
                        candidates.append(str(v[k]))

    return list(dict.fromkeys(candidates))


def extract_names_from_result(result: Any) -> List[str]:
    names: List[str] = []
    if not isinstance(result, list):
        return names

    for row in result:
        if not isinstance(row, dict):
            continue

        for k in NAME_KEYS:
            if k in row and row[k] is not None:
                names.append(str(row[k]))

        for _, v in row.items():
            if isinstance(v, dict):
                for k in NAME_KEYS:
                    if k in v and v[k] is not None:
                        names.append(str(v[k]))

    names = list(dict.fromkeys(names))
    return names[:20]


# ----------------------------
# Main builder
# ----------------------------
def build_langgraph_app(graph: Neo4jGraph, llm: ChatOllama):

    # 1) generate cypher
    def node_generate_cypher(state: GraphState) -> GraphState:
        schema = state["schema"]
        question = state["question"]

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert Neo4j Cypher generator.\n"
             "Rules:\n"
             "1) Generate ONE read-only Cypher query.\n"
             "2) NEVER use '.elementId'. Use elementId(var) function.\n"
             "3) When possible, RETURN elementId(mainNode) AS elementId and coalesce(mainNode.Name, mainNode.name, mainNode.title) AS name.\n"
             "4) Keep results small with LIMIT 25 (unless already smaller).\n"
             "5) Return ONLY Cypher text. No markdown.\n"
             "6) No CREATE/MERGE/DELETE/SET.\n"),
            ("human", "Schema:\n{schema}\n\nQuestion:\n{question}\n\nCypher:")
        ])

        cypher = llm.invoke(prompt.format(schema=schema, question=question)).content.strip()

        # normalize elementId usage
        cypher = normalize_elementid_usage(cypher)

        # force LIMIT if missing
        if "limit" not in cypher.lower():
            cypher = cypher.rstrip(";") + "\nLIMIT 25"

        if not is_safe_readonly_cypher(cypher):
            return {**state, "cypher": cypher, "error": f"Unsafe/invalid Cypher generated:\n{cypher}"}

        return {**state, "cypher": cypher}

    # 2) run cypher
    def node_run_cypher(state: GraphState) -> GraphState:
        if state.get("error"):
            return state
        try:
            result = graph.query(state["cypher"])
            return {**state, "result": result}
        except Exception as e:
            return {**state, "error": f"Neo4j query error: {e}"}

    # 3) build subgraph (1-hop or 2-hop)
    def node_build_subgraph(state: GraphState) -> GraphState:
        if state.get("error"):
            return state

        hop = int(state.get("hop", 1))
        hop = 2 if hop == 2 else 1

        result = state.get("result", [])

        # (A) primary: elementId-based
        eids = extract_eids_from_result(result)[:20]

        # (B) fallback: Name-based -> find elementId
        if not eids:
            names = extract_names_from_result(result)
            if names:
                cypher_find = """
                UNWIND $names AS nm
                MATCH (n)
                WHERE coalesce(n.Name, n.name, n.title) = nm
                RETURN elementId(n) AS eid
                LIMIT 50
                """
                try:
                    rows = graph.query(cypher_find, params={"names": names})
                except TypeError:
                    rows = graph.query(cypher_find, {"names": names})

                eids = [str(r["eid"]) for r in rows if isinstance(r, dict) and r.get("eid")]
                eids = list(dict.fromkeys(eids))[:20]

        if not eids:
            return {**state, "subgraph": {"nodes": [], "edges": []}}

        # ✅ hop별 Cypher
        if hop == 1:
            # seed에서 Wage 제외
            cypher_sg = """
            UNWIND $eids AS eid
            MATCH (n) WHERE elementId(n) = eid AND NOT n:Wage
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN
              elementId(n) AS src_id,
              labels(n) AS src_labels,
              properties(n) AS src_props,
              coalesce(n.Name, n.name, n.title) AS src_name,
              elementId(m) AS dst_id,
              labels(m) AS dst_labels,
              properties(m) AS dst_props,
              coalesce(m.Name, m.name, m.title) AS dst_name,
              type(r) AS rel_type
            LIMIT 600
            """
        else:
            # 2-hop: seed에서 Wage 제외, 그리고 2-hop 경로 확장
            # p=(n)-[r1]-(x)-[r2]-(m) 형태로 노드/관계 수집
            cypher_sg = """
            UNWIND $eids AS eid
            MATCH (n) WHERE elementId(n) = eid AND NOT n:Wage
            MATCH (n)-[r1]-(x)
            OPTIONAL MATCH (x)-[r2]-(m)
            RETURN
              elementId(n) AS n_id, labels(n) AS n_labels, properties(n) AS n_props, coalesce(n.Name, n.name, n.title) AS n_name,
              elementId(x) AS x_id, labels(x) AS x_labels, properties(x) AS x_props, coalesce(x.Name, x.name, x.title) AS x_name,
              type(r1) AS r1_type,
              elementId(m) AS m_id, labels(m) AS m_labels, properties(m) AS m_props, coalesce(m.Name, m.name, m.title) AS m_name,
              type(r2) AS r2_type
            LIMIT 900
            """

        try:
            sg = graph.query(cypher_sg, params={"eids": eids})
        except TypeError:
            sg = graph.query(cypher_sg, {"eids": eids})

        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, str]] = []

        def upsert(node_id: str, labels: List[str], props: Dict[str, Any], name: Any):
            if not node_id or node_id in nodes:
                return
            nodes[node_id] = {
                "id": node_id,
                "type": ":".join(labels or []),
                "name": (name or props.get("Name") or props.get("name") or props.get("title")),
                "props": props or {},
            }

        if hop == 1:
            for row in sg:
                src_id = row.get("src_id")
                dst_id = row.get("dst_id")
                rel_type = row.get("rel_type")

                upsert(src_id, row.get("src_labels") or [], row.get("src_props") or {}, row.get("src_name"))
                upsert(dst_id, row.get("dst_labels") or [], row.get("dst_props") or {}, row.get("dst_name"))

                if src_id and dst_id and rel_type:
                    edges.append({"source": src_id, "target": dst_id, "label": rel_type})
        else:
            for row in sg:
                n_id = row.get("n_id")
                x_id = row.get("x_id")
                m_id = row.get("m_id")

                upsert(n_id, row.get("n_labels") or [], row.get("n_props") or {}, row.get("n_name"))
                upsert(x_id, row.get("x_labels") or [], row.get("x_props") or {}, row.get("x_name"))
                if m_id:
                    upsert(m_id, row.get("m_labels") or [], row.get("m_props") or {}, row.get("m_name"))

                r1 = row.get("r1_type")
                r2 = row.get("r2_type")

                if n_id and x_id and r1:
                    edges.append({"source": n_id, "target": x_id, "label": r1})
                if x_id and m_id and r2:
                    edges.append({"source": x_id, "target": m_id, "label": r2})

        # edges 중복 제거
        seen = set()
        dedup_edges = []
        for e in edges:
            key = (e["source"], e["target"], e.get("label", ""))
            if key not in seen:
                seen.add(key)
                dedup_edges.append(e)

        return {**state, "subgraph": {"nodes": list(nodes.values()), "edges": dedup_edges}}

    # 4) summarize
    def node_summarize(state: GraphState) -> GraphState:
        if state.get("error"):
            return state

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You summarize Neo4j query results for the user in Korean. Be concise and clear."),
            ("human", "Question:\n{question}\n\nCypher:\n{cypher}\n\nResult:\n{result}\n\nAnswer:")
        ])
        answer = llm.invoke(prompt.format(
            question=state["question"],
            cypher=state["cypher"],
            result=state.get("result", [])
        )).content.strip()

        return {**state, "answer": answer}

    workflow = StateGraph(GraphState)
    workflow.add_node("generate_cypher", node_generate_cypher)
    workflow.add_node("run_cypher", node_run_cypher)
    workflow.add_node("build_subgraph", node_build_subgraph)
    workflow.add_node("summarize", node_summarize)

    workflow.set_entry_point("generate_cypher")
    workflow.add_edge("generate_cypher", "run_cypher")
    workflow.add_edge("run_cypher", "build_subgraph")
    workflow.add_edge("build_subgraph", "summarize")
    workflow.add_edge("summarize", END)

    app = workflow.compile()

    def invoke(question: str, schema: str, hop: int = 1) -> GraphState:
        return app.invoke({"question": question, "schema": schema, "hop": hop})

    return invoke
