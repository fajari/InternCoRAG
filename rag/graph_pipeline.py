from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Any, Dict, List, Tuple

try:
    import networkx as nx
except Exception:
    nx = None

try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None

try:
    from llama_index.core import Document as LlamaDocument
    from llama_index.core import KnowledgeGraphIndex, StorageContext
    from llama_index.core.graph_stores import SimpleGraphStore
except Exception:
    LlamaDocument = None
    KnowledgeGraphIndex = None
    StorageContext = None
    SimpleGraphStore = None

try:
    import graphrag  # noqa: F401
except Exception:
    graphrag = None


@dataclass
class GraphBundle:
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    communities: List[Dict[str, Any]] = field(default_factory=list)
    central_nodes: List[str] = field(default_factory=list)
    kg_triplets: List[Tuple[str, str, str]] = field(default_factory=list)
    local_search_lines: List[str] = field(default_factory=list)
    capabilities: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    networkx_graph: Any | None = None
    neo4j_synced: bool = False
    cypher_path_lines: List[str] = field(default_factory=list)


def detect_graph_capabilities() -> Dict[str, bool]:
    return {
        "networkx": nx is not None,
        "neo4j_driver": GraphDatabase is not None,
        "llamaindex_kg_index": KnowledgeGraphIndex is not None,
        "microsoft_graphrag": graphrag is not None,
    }


def edge_category(relation: str) -> str:
    relation_norm = (relation or "").strip().lower()
    if relation_norm in {"mentransfer", "membayar", "menerima"}:
        return "aliran_dana"
    if relation_norm in {"menghubungi", "bertemu", "berkomunikasi"}:
        return "komunikasi"
    if relation_norm in {"melaporkan", "menuduh", "memeriksa", "menyetujui", "memerintahkan"}:
        return "proses_hukum"
    return "relasi"


def build_networkx_graph(
    actors: List[str],
    relationships: List[tuple[str, str, str, str]],
    actor_profiles: Dict[str, Dict[str, Any]],
) -> Any | None:
    if nx is None:
        return None

    graph = nx.MultiDiGraph()

    for actor in actors:
        profile = actor_profiles.get(actor, {})
        graph.add_node(
            actor,
            role=profile.get("role", "Pihak Lain"),
            context=profile.get("context", ""),
        )

    for left, relation, right, evidence in relationships:
        if left not in graph:
            graph.add_node(left, role=actor_profiles.get(left, {}).get("role", "Pihak Lain"))
        if right not in graph:
            graph.add_node(right, role=actor_profiles.get(right, {}).get("role", "Pihak Lain"))
        graph.add_edge(
            left,
            right,
            relation=relation,
            evidence=evidence,
            category=edge_category(relation),
        )

    return graph


def build_central_nodes(graph: Any | None, limit: int = 5) -> List[str]:
    if nx is None or graph is None or graph.number_of_nodes() == 0:
        return []

    undirected = graph.to_undirected()
    degree = nx.degree_centrality(undirected)
    betweenness = nx.betweenness_centrality(undirected) if undirected.number_of_edges() else {}
    ranked = []
    for node in undirected.nodes:
        score = degree.get(node, 0.0) + (betweenness.get(node, 0.0) * 1.5)
        ranked.append((score, str(node)))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [node for _, node in ranked[:limit]]


def build_communities(graph: Any | None, actor_profiles: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if nx is None or graph is None or graph.number_of_nodes() < 3:
        return []

    undirected = graph.to_undirected()
    if undirected.number_of_edges() == 0:
        return []

    communities = []
    try:
        detected = list(nx.algorithms.community.greedy_modularity_communities(undirected))
    except Exception:
        return []

    for index, community in enumerate(detected, start=1):
        members = sorted(str(node) for node in community)
        roles = []
        for member in members:
            role = actor_profiles.get(member, {}).get("role", "Pihak Lain")
            roles.append(f"{member} [{role}]")
        communities.append(
            {
                "id": index,
                "members": members,
                "summary": f"Klaster {index} berisi {'; '.join(roles[:6])}.",
            }
        )

    return communities


def build_local_search_lines(
    question: str,
    graph: Any | None,
    actor_profiles: Dict[str, Dict[str, Any]],
    limit: int = 6,
) -> List[str]:
    if graph is None or graph.number_of_nodes() == 0:
        return []

    query = (question or "").lower()
    matched_nodes = [
        node for node in graph.nodes
        if str(node).lower() in query or any(token in str(node).lower() for token in query.split())
    ]
    if not matched_nodes:
        matched_nodes = build_central_nodes(graph, limit=3)

    lines: List[str] = []
    seen = set()
    for node in matched_nodes:
        role = actor_profiles.get(str(node), {}).get("role", "Pihak Lain")
        for _, target, payload in graph.out_edges(node, data=True):
            relation = payload.get("relation", "terkait")
            key = (str(node), relation, str(target))
            if key in seen:
                continue
            seen.add(key)
            target_role = actor_profiles.get(str(target), {}).get("role", "Pihak Lain")
            lines.append(f"- {node} [{role}] -> {relation} -> {target} [{target_role}]")
            if len(lines) >= limit:
                return lines

    return lines[:limit]


def build_llamaindex_triplets(semantic_chunks: List[str], max_chunks: int = 8) -> List[Tuple[str, str, str]]:
    if not (KnowledgeGraphIndex and LlamaDocument and StorageContext and SimpleGraphStore):
        return []

    enabled = os.getenv("ENABLE_LLAMAINDEX_KG", "false").lower() == "true"
    if not enabled:
        return []

    try:
        documents = [
            LlamaDocument(text=chunk)
            for chunk in semantic_chunks[:max_chunks]
            if chunk and chunk.strip()
        ]
        if not documents:
            return []

        storage_context = StorageContext.from_defaults(graph_store=SimpleGraphStore())
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            max_triplets_per_chunk=8,
            show_progress=False,
        )
        triplets = []
        graph_store = storage_context.graph_store
        graph_dict = getattr(graph_store, "_data", {}) or {}
        for subject, relations in graph_dict.items():
            for relation, target in relations:
                triplets.append((str(subject), str(relation), str(target)))
        return triplets[:32]
    except Exception:
        return []


def sync_graph_to_neo4j(bundle: GraphBundle, workspace_id: str) -> bool:
    if GraphDatabase is None:
        return False

    enabled = os.getenv("ENABLE_NEO4J_SYNC", "false").lower() == "true"
    if not enabled:
        return False

    uri = os.getenv("NEO4J_URI", "").strip()
    username = os.getenv("NEO4J_USERNAME", "").strip()
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip()
    if not (uri and username and password):
        return False

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session(database=database) as session:
            session.run(
            """
            MERGE (w:Workspace {id: $workspace_id})
            """,
                workspace_id=workspace_id,
            )
            for node in bundle.nodes:
                session.run(
                """
                MERGE (p:Person {workspace_id: $workspace_id, name: $name})
                SET p.role = $role, p.context = $context
                """,
                    workspace_id=workspace_id,
                    name=node.get("name", ""),
                    role=node.get("role", "Pihak Lain"),
                    context=node.get("context", ""),
                )
            for edge in bundle.edges:
                session.run(
                """
                MATCH (a:Person {workspace_id: $workspace_id, name: $left})
                MATCH (b:Person {workspace_id: $workspace_id, name: $right})
                MERGE (a)-[r:RELATED {relation: $relation}]->(b)
                SET r.evidence = $evidence, r.category = $category
                """,
                    workspace_id=workspace_id,
                    left=edge.get("left", ""),
                    right=edge.get("right", ""),
                    relation=edge.get("relation", "terkait"),
                    evidence=edge.get("evidence", ""),
                    category=edge.get("category", "relasi"),
                )
        driver.close()
        return True
    except Exception:
        return False


def get_neo4j_graph_connection() -> Any | None:
    if GraphDatabase is None:
        return None

    enabled = os.getenv("ENABLE_NEO4J_SYNC", "false").lower() == "true"
    if not enabled:
        return None

    uri = os.getenv("NEO4J_URI", "").strip()
    username = os.getenv("NEO4J_USERNAME", "").strip()
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip()
    if not (uri and username and password):
        return None

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        return {
            "driver": driver,
            "database": database,
        }
    except Exception:
        return None


def extract_question_actors(question: str, node_names: List[str]) -> List[str]:
    question_norm = re.sub(r"\s+", " ", (question or "").lower()).strip()
    matched = []
    for node_name in node_names:
        node_norm = re.sub(r"\s+", " ", str(node_name).lower()).strip()
        if node_norm and node_norm in question_norm:
            matched.append(str(node_name))
    return matched[:4]


def is_cross_relation_question(question: str, graph_bundle: GraphBundle) -> bool:
    question_norm = (question or "").lower()
    relation_markers = (
        "hubungan", "relasi", "jalur", "path", "lintas relasi",
        "terhubung", "koneksi", "antara", "siapa terkait",
        "jaringan", "melalui siapa", "hubungkan",
    )
    matched_actors = extract_question_actors(question, [node.get("name", "") for node in graph_bundle.nodes])
    return any(marker in question_norm for marker in relation_markers) or len(matched_actors) >= 2


def format_path(nodes: List[str], relations: List[str]) -> str:
    if not nodes:
        return ""
    if not relations:
        return " -> ".join(nodes)
    parts = [nodes[0]]
    for index, relation in enumerate(relations):
        if index + 1 >= len(nodes):
            break
        parts.append(relation)
        parts.append(nodes[index + 1])
    return " -> ".join(part for part in parts if part)


def query_neo4j_relationship_paths(
    workspace_id: str,
    question: str,
    graph_bundle: GraphBundle,
    max_hops: int = 4,
) -> List[str]:
    graph = get_neo4j_graph_connection()
    if graph is None:
        return []
    driver = graph["driver"]
    database = graph["database"]

    matched_actors = extract_question_actors(question, [node.get("name", "") for node in graph_bundle.nodes])
    lines: List[str] = []

    try:
        with driver.session(database=database) as session:
            if len(matched_actors) >= 2:
                result = session.run(
                f"""
                MATCH p = shortestPath(
                    (a:Person {{workspace_id: $workspace_id, name: $start}})
                    -[:RELATED*..{max_hops}]-
                    (b:Person {{workspace_id: $workspace_id, name: $end}})
                )
                RETURN [n IN nodes(p) | n.name] AS nodes,
                       [r IN relationships(p) | r.relation] AS relations
                LIMIT 3
                """,
                    workspace_id=workspace_id,
                    start=matched_actors[0],
                    end=matched_actors[1],
                )
                for row in result:
                    path_text = format_path(row.get("nodes") or [], row.get("relations") or [])
                    if path_text:
                        lines.append(f"- {path_text}")
                if lines:
                    driver.close()
                    return lines

            for actor in matched_actors[:2]:
                result = session.run(
                """
                MATCH (a:Person {workspace_id: $workspace_id, name: $actor})-[r:RELATED]-(b:Person {workspace_id: $workspace_id})
                RETURN a.name AS left_name, r.relation AS relation, b.name AS right_name, r.category AS category
                LIMIT 8
                """,
                    workspace_id=workspace_id,
                    actor=actor,
                )
                for row in result:
                    left_name = row.get("left_name", actor)
                    relation = row.get("relation", "terkait")
                    right_name = row.get("right_name", "")
                    category = row.get("category", "relasi")
                    if right_name:
                        lines.append(f"- {left_name} -> {relation} -> {right_name} [{category}]")
                if lines:
                    driver.close()
                    return lines[:6]
    except Exception:
        try:
            driver.close()
        except Exception:
            pass
        return []

    driver.close()
    return []


def query_networkx_relationship_paths(
    question: str,
    graph_bundle: GraphBundle,
    max_hops: int = 4,
) -> List[str]:
    if nx is None or graph_bundle.networkx_graph is None:
        return []

    matched_actors = extract_question_actors(question, [node.get("name", "") for node in graph_bundle.nodes])
    if not matched_actors:
        return graph_bundle.local_search_lines[:4]

    graph = graph_bundle.networkx_graph
    undirected = graph.to_undirected()
    lines: List[str] = []

    if len(matched_actors) >= 2:
        start, end = matched_actors[0], matched_actors[1]
        try:
            path = nx.shortest_path(undirected, source=start, target=end)
            if len(path) - 1 <= max_hops:
                relations: List[str] = []
                for left, right in zip(path, path[1:]):
                    edge_data = graph.get_edge_data(left, right) or graph.get_edge_data(right, left) or {}
                    first_payload = next(iter(edge_data.values()), {})
                    relations.append(first_payload.get("relation", "terkait"))
                lines.append(f"- {format_path(path, relations)}")
        except Exception:
            pass

    if lines:
        return lines

    for actor in matched_actors[:2]:
        role = next((node.get("role", "Pihak Lain") for node in graph_bundle.nodes if node.get("name") == actor), "Pihak Lain")
        for _, target, payload in graph.out_edges(actor, data=True):
            relation = payload.get("relation", "terkait")
            category = payload.get("category", "relasi")
            lines.append(f"- {actor} [{role}] -> {relation} -> {target} [{category}]")
            if len(lines) >= 6:
                return lines

    return lines[:6]


def resolve_relationship_paths(
    workspace_id: str,
    question: str,
    graph_bundle: GraphBundle,
) -> List[str]:
    lines = query_neo4j_relationship_paths(workspace_id, question, graph_bundle)
    if lines:
        return lines
    return query_networkx_relationship_paths(question, graph_bundle)


def build_graph_bundle(
    workspace_id: str,
    question: str,
    semantic_chunks: List[str],
    actors: List[str],
    relationships: List[tuple[str, str, str, str]],
    actor_profiles: Dict[str, Dict[str, Any]],
) -> GraphBundle:
    capabilities = detect_graph_capabilities()
    nodes = [
        {
            "name": actor,
            "role": actor_profiles.get(actor, {}).get("role", "Pihak Lain"),
            "context": actor_profiles.get(actor, {}).get("context", ""),
        }
        for actor in actors
    ]
    edges = [
        {
            "left": left,
            "relation": relation,
            "right": right,
            "evidence": evidence,
            "category": edge_category(relation),
        }
        for left, relation, right, evidence in relationships
    ]

    graph = build_networkx_graph(actors, relationships, actor_profiles)
    communities = build_communities(graph, actor_profiles)
    central_nodes = build_central_nodes(graph)
    local_search_lines = build_local_search_lines(question, graph, actor_profiles)
    kg_triplets = build_llamaindex_triplets(semantic_chunks)

    bundle = GraphBundle(
        nodes=nodes,
        edges=edges,
        communities=communities,
        central_nodes=central_nodes,
        kg_triplets=kg_triplets,
        local_search_lines=local_search_lines,
        capabilities=capabilities,
        networkx_graph=graph,
    )
    bundle.neo4j_synced = sync_graph_to_neo4j(bundle, workspace_id)
    bundle.cypher_path_lines = resolve_relationship_paths(workspace_id, question, bundle)

    if not capabilities.get("microsoft_graphrag"):
        bundle.warnings.append(
            "Microsoft GraphRAG belum terpasang di environment ini; pipeline graph lokal dipakai sebagai fallback."
        )
    if not capabilities.get("llamaindex_kg_index"):
        bundle.warnings.append(
            "LlamaIndex KG Index belum tersedia; ekstraksi triplet tambahan tidak dijalankan."
        )
    if capabilities.get("neo4j_driver") and not bundle.neo4j_synced and os.getenv("ENABLE_NEO4J_SYNC", "false").lower() == "true":
        bundle.warnings.append(
            "Sinkronisasi Neo4j diaktifkan tetapi belum berhasil; periksa kredensial Neo4j."
        )

    return bundle
