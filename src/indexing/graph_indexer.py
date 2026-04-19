"""
Neo4j-backed Knowledge Graph Indexer.

Graph schema:
  (:Candidate)-[:HAS_SKILL]->(:Skill)
  (:Candidate)-[:HAS_ROLE]->(:Role)

This enables 2-hop traversal paths like:
  Query → [mongodb] → [Backend Developer] → Candidate

Connection defaults (override with env vars):
  NEO4J_URI      (default: bolt://localhost:7687)
  NEO4J_USER     (default: neo4j)
  NEO4J_PASSWORD (default: test1234)
"""
import os
import re as _re
from typing import List, Dict, Any
from neo4j import GraphDatabase
from src.models.schema import CandidateProfile


NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Akshat007")


class GraphIndexer:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.profiles: List[CandidateProfile] = []
        self._profile_map: Dict[str, CandidateProfile] = {}
        
        # State for semantic node matching
        self.node_names: List[str] = []
        self.semantic_model = None
        self.node_embeddings = None

    def close(self):
        self.driver.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        """Split 'Python (Advanced), Java (Beginner)' → ['python', 'java']"""
        if not text:
            return []
        result = []
        for part in text.split(","):
            clean = part.split("(")[0].strip().lower()
            if clean:
                result.append(clean)
        return result

    # ------------------------------------------------------------------
    # Indexing: Candidate → Skill, Candidate → Role
    # ------------------------------------------------------------------
    def index(self, profiles: List[CandidateProfile]):
        self.profiles = profiles
        self._profile_map = {p.id: p for p in profiles}
        
        all_unique_nodes = set()

        with self.driver.session() as session:
            # Clean slate
            session.run("MATCH (n) DETACH DELETE n")

            # Uniqueness constraints
            for constraint in [
                "CREATE CONSTRAINT candidate_id IF NOT EXISTS FOR (c:Candidate) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT skill_name   IF NOT EXISTS FOR (s:Skill)     REQUIRE s.name IS UNIQUE",
                "CREATE CONSTRAINT role_name    IF NOT EXISTS FOR (r:Role)      REQUIRE r.name IS UNIQUE",
            ]:
                try:
                    session.run(constraint)
                except Exception:
                    pass

            def _index_profile(tx, p):
                # ── Entity node ────────────────────────────────────
                tx.run(
                    f"""
                    MERGE (c:Candidate {{id: $id}})
                    SET c.name = $name,
                        c.canonical_id = $canonical_id,
                        c.entity_type = $entity_type
                    """,
                    id=p.id,
                    name=p.name or "",
                    canonical_id=p.canonical_id or "",
                    entity_type=p.entity_type or "candidate",
                )

                # ── Dynamic Attributes ────────────────────────────────
                for attr in p.attributes:
                    if attr.value is None:
                        continue
                    
                    vals = attr.value if isinstance(attr.value, list) else [attr.value]
                    label_str = attr.key.replace(" ", "").title()
                    rel_type = f"HAS_{attr.key.replace(' ', '_').upper()}"
                    
                    for v in vals:
                        clean_v = str(v).strip()
                        if not clean_v: continue
                        
                        tx.run(
                            f"""
                            MERGE (n:{label_str} {{name: $val}})
                            WITH n
                            MATCH (c:Candidate {{id: $id}})
                            MERGE (c)-[:{rel_type}]->(n)
                            """,
                            val=clean_v,
                            id=p.id,
                        )
                        all_unique_nodes.add(clean_v)

                # ── First-Class Relations ─────────────────────────────
                for rel in p.relations:
                    rel_label = rel.relation_type.replace(" ", "_").upper()
                    target_label = (rel.target_type or "Entity").replace(" ", "").title()
                    
                    tx.run(
                        f"""
                        MERGE (t:{target_label} {{id: $t_id}})
                        WITH t
                        MATCH (s:Candidate {{id: $id}})
                        MERGE (s)-[:{rel_label}]->(t)
                        """,
                        t_id=rel.target_id,
                        id=p.id,
                    )
                    # We might not have a clean 'name' for relations to do semantic routing against easily,
                    # but if target_id is descriptive, we add it
                    all_unique_nodes.add(rel.target_id)

            for p in profiles:
                try:
                    session.execute_write(_index_profile, p)
                except Exception as e:
                    print(f"[GraphIndexer] Warning: Failed to index profile {p.id}: {e}")
                    
        # --- Compute Semantic Embeddings for Graph Nodes ---
        self.node_names = list(all_unique_nodes)
        if self.node_names:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.node_embeddings = self.semantic_model.encode(self.node_names, convert_to_tensor=True)
                print(f"[GraphIndexer] Computed embeddings for {len(self.node_names)} graph nodes for semantic proxy routing.")
            except Exception as e:
                print(f"[GraphIndexer] Warning: Failed to load semantic fallback model ({e})")

        print(f"[GraphIndexer] Indexed {len(profiles)} candidates dynamically into Neo4j.")

    # ------------------------------------------------------------------
    # Incremental single-profile add (no full wipe)
    # ------------------------------------------------------------------
    def index_single(self, profile: CandidateProfile):
        """
        MERGE a single new candidate + its attribute nodes into Neo4j.
        Also extends the in-memory semantic node embeddings so proxy
        routing works for the new profile's skills/roles immediately.
        No full graph wipe — safe to call on a live running system.
        """
        self.profiles.append(profile)
        self._profile_map[profile.id] = profile

        new_node_values: set = set()

        def _index_single_tx(tx):
            # MERGE candidate node
            tx.run(
                """
                MERGE (c:Candidate {id: $id})
                SET c.name        = $name,
                    c.canonical_id = $canonical_id,
                    c.entity_type  = $entity_type
                """,
                id=profile.id,
                name=profile.name or "",
                canonical_id=profile.canonical_id or "",
                entity_type=profile.entity_type or "candidate",
            )

            # MERGE attribute nodes + relationships
            for attr in profile.attributes:
                if attr.value is None:
                    continue
                vals = attr.value if isinstance(attr.value, list) else [attr.value]
                label_str = attr.key.replace(" ", "").title()
                rel_type = f"HAS_{attr.key.replace(' ', '_').upper()}"

                for v in vals:
                    clean_v = str(v).strip()
                    if not clean_v:
                        continue
                    tx.run(
                        f"""
                        MERGE (n:{label_str} {{name: $val}})
                        WITH n
                        MATCH (c:Candidate {{id: $id}})
                        MERGE (c)-[:{rel_type}]->(n)
                        """,
                        val=clean_v,
                        id=profile.id,
                    )
                    new_node_values.add(clean_v)

        with self.driver.session() as session:
            # Ensure constraints exist (idempotent)
            for constraint in [
                "CREATE CONSTRAINT candidate_id IF NOT EXISTS FOR (c:Candidate) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT skill_name   IF NOT EXISTS FOR (s:Skill)     REQUIRE s.name IS UNIQUE",
                "CREATE CONSTRAINT role_name    IF NOT EXISTS FOR (r:Role)      REQUIRE r.name IS UNIQUE",
            ]:
                try:
                    session.run(constraint)
                except Exception:
                    pass

            session.execute_write(_index_single_tx)

        # Extend in-memory semantic node embeddings for new nodes only
        if new_node_values and self.semantic_model is not None:
            try:
                import torch
                existing = set(self.node_names)
                truly_new = [n for n in new_node_values if n not in existing]
                if truly_new:
                    new_embs = self.semantic_model.encode(truly_new, convert_to_tensor=True)
                    self.node_names.extend(truly_new)
                    if self.node_embeddings is not None:
                        self.node_embeddings = torch.cat([self.node_embeddings, new_embs], dim=0)
                    else:
                        self.node_embeddings = new_embs
            except Exception as e:
                print(f"[GraphIndexer] Warning: Could not extend node embeddings: {e}")

        print(f"[GraphIndexer] Incrementally indexed candidate {profile.id} into Neo4j.")

    # ------------------------------------------------------------------
    # Update single profile in graph (delete old relationships, add new)
    # ------------------------------------------------------------------
    def update_single(self, profile: CandidateProfile):
        """
        Update a candidate already in Neo4j:
          1. Delete all outgoing attribute relationships from the candidate node.
          2. Re-MERGE all new attribute nodes and relationships.
          3. Update in-memory profile map and semantic node embeddings.
        Safe to call on a live system — no full graph wipe.
        """
        # Update in-memory profile map
        if profile.id in self._profile_map:
            for i, p in enumerate(self.profiles):
                if p.id == profile.id:
                    self.profiles[i] = profile
                    break
        else:
            self.profiles.append(profile)
        self._profile_map[profile.id] = profile

        new_node_values: set = set()

        def _update_single_tx(tx):
            # Delete existing outgoing relationships (keeps the Candidate node)
            tx.run(
                "MATCH (c:Candidate {id: $id})-[r]->() DELETE r",
                id=profile.id,
            )

            # Re-MERGE candidate node properties
            tx.run(
                """
                MERGE (c:Candidate {id: $id})
                SET c.name         = $name,
                    c.canonical_id  = $canonical_id,
                    c.entity_type   = $entity_type
                """,
                id=profile.id,
                name=profile.name or "",
                canonical_id=profile.canonical_id or "",
                entity_type=profile.entity_type or "candidate",
            )

            # MERGE new attribute nodes + relationships
            for attr in profile.attributes:
                if attr.value is None:
                    continue
                vals = attr.value if isinstance(attr.value, list) else [attr.value]
                label_str = attr.key.replace(" ", "").title()
                rel_type = f"HAS_{attr.key.replace(' ', '_').upper()}"
                for v in vals:
                    clean_v = str(v).strip()
                    if not clean_v:
                        continue
                    tx.run(
                        f"""
                        MERGE (n:{label_str} {{name: $val}})
                        WITH n
                        MATCH (c:Candidate {{id: $id}})
                        MERGE (c)-[:{rel_type}]->(n)
                        """,
                        val=clean_v,
                        id=profile.id,
                    )
                    new_node_values.add(clean_v)

        with self.driver.session() as session:
            session.execute_write(_update_single_tx)

        # Extend in-memory semantic embeddings for any truly new attribute values
        if new_node_values and self.semantic_model is not None:
            try:
                import torch
                existing = set(self.node_names)
                truly_new = [n for n in new_node_values if n not in existing]
                if truly_new:
                    new_embs = self.semantic_model.encode(truly_new, convert_to_tensor=True)
                    self.node_names.extend(truly_new)
                    if self.node_embeddings is not None:
                        self.node_embeddings = torch.cat([self.node_embeddings, new_embs], dim=0)
                    else:
                        self.node_embeddings = new_embs
            except Exception as e:
                print(f"[GraphIndexer] Warning: Could not extend node embeddings: {e}")

        print(f"[GraphIndexer] Updated candidate {profile.id} in Neo4j.")

    # ------------------------------------------------------------------
    # Search: Agnistic Multi-Hop Traversal 
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Phase 1 — direct node match:
          Query tokens → matched dynamic nodes → connected Candidates

        Phase 2 — indirect multi-hop generic traversal:
          Find candidates connected to ANY intermediate shared graph node
        """
        tokens = [t.strip() for t in query.lower().split() if len(t.strip()) > 2]
        seen: set = set()
        unique_tokens = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)
        tokens = unique_tokens

        if not tokens:
            return []

        escaped = [_re.escape(t) for t in tokens]
        regex_parts = escaped.copy()
        semantic_proxies = {}

        if self.semantic_model is not None and self.node_embeddings is not None:
            import torch
            from sentence_transformers import util
            for t in tokens:
                t_lower = t.lower()
                exact_match = any(t_lower in n.lower() for n in self.node_names)
                if exact_match:
                    continue
                
                with torch.no_grad():
                    t_emb = self.semantic_model.encode(t, convert_to_tensor=True)
                    sims = util.cos_sim(t_emb, self.node_embeddings)[0]
                    best_idx = torch.argmax(sims).item()
                    best_score = sims[best_idx].item()
                    
                    if best_score > 0.4:
                        best_node = self.node_names[best_idx]
                        regex_parts.append(_re.escape(best_node))
                        semantic_proxies[best_node] = t

        regex = "(?i)(" + "|".join(regex_parts) + ")"

        with self.driver.session() as session:
            # ── Phase 1: Relation-Agnostic Direct Links ────────
            direct_results = session.run(
                """
                MATCH (s)
                WHERE s.name =~ $regex OR s.id =~ $regex
                MATCH (c:Candidate)-[]->(s)
                RETURN c.id AS cid, coalesce(s.name, s.id) AS skill_name
                """,
                regex=regex,
            ).data()

            # ── Phase 2: Relation-Agnostic Indirect Multi-Hop ────
            indirect_results = session.run(
                """
                MATCH (s) WHERE s.name =~ $regex OR s.id =~ $regex
                MATCH (c_dir:Candidate)-[]->(s)
                MATCH (c_dir)-[]->(intermediate)<-[]-(c_indir:Candidate)
                WHERE NOT (c_indir)-[]->(s)
                RETURN c_indir.id AS cid, coalesce(s.name, s.id) AS skill_name, coalesce(intermediate.name, intermediate.id, 'Node') AS role_name
                LIMIT 500
                """,
                regex=regex,
            ).data()

        # Score candidates and build their traversal segments
        candidate_scores = {}
        candidate_segments = {}

        # 1. Process direct matches
        for row in direct_results:
            cid = row["cid"]
            skill = row["skill_name"]
            
            if cid not in candidate_scores:
                candidate_scores[cid] = 0.0
                candidate_segments[cid] = {}
            
            candidate_scores[cid] += 1.0  # Direct matches get full score
            candidate_segments[cid][skill] = {
                "skill": skill,
                "semantic_proxy": semantic_proxies.get(skill),
                "via_roles": [],
                "direct": True
            }

        # 2. Process indirect matches
        for row in indirect_results:
            cid = row["cid"]
            skill = row["skill_name"]
            role = row["role_name"]

            if cid not in candidate_scores:
                candidate_scores[cid] = 0.0
                candidate_segments[cid] = {}
            
            if skill not in candidate_segments[cid]:
                candidate_segments[cid][skill] = {
                    "skill": skill,
                    "semantic_proxy": semantic_proxies.get(skill),
                    "via_roles": set(),
                    "direct": False
                }
            
            # Only add via_roles if it wasn't already a direct match
            if not candidate_segments[cid][skill]["direct"]:
                candidate_segments[cid][skill]["via_roles"].add(role)

        # 3. Finalize scores and build result objects
        for cid, segs in candidate_segments.items():
            for seg in segs.values():
                if not seg["direct"]:
                    seg["via_roles"] = list(seg["via_roles"])[:2]
                    candidate_scores[cid] += 0.5  # Indirect matches get half score

        # Sort candidates by score
        sorted_cids = sorted(candidate_scores.keys(), key=lambda x: candidate_scores[x], reverse=True)[:top_k]
        max_score = max(candidate_scores.values()) if candidate_scores else 1.0

        results = []
        for cid in sorted_cids:
            profile = self._profile_map.get(cid)
            if profile is None:
                continue

            segs = list(candidate_segments[cid].values())
            raw_score = candidate_scores[cid]
            norm_score = raw_score / max(max_score, 1.0)

            # Flat traversal list for backward compatibility
            flat_skills = [s["skill"] for s in segs]

            results.append({
                "profile": profile,
                "score": float(norm_score),
                "technique": "graph",
                "graph_traversal": flat_skills,
                "graph_traversal_detail": segs,
            })
            
        return results

    # ------------------------------------------------------------------
    # Graph data for visualization (unchanged)
    # ------------------------------------------------------------------
    def get_graph_data(self, max_skills: int = 60, max_candidates: int = 80, search_term: str = None) -> Dict[str, Any]:
        """Return graph data as nodes/edges dict for D3.js visualization dynamically."""
        with self.driver.session() as session:
            if search_term:
                search_term = search_term.lower()
                top_skills_result = session.run(
                    """
                    MATCH (c:Candidate)-[]->(s)
                    WHERE NOT 'Candidate' IN labels(s)
                      AND (toLower(s.name) CONTAINS $search_term OR toLower(coalesce(c.name, '')) CONTAINS $search_term)
                    WITH s, count(c) AS degree
                    ORDER BY degree DESC
                    LIMIT $max_skills
                    RETURN coalesce(s.name, s.id) AS name, degree
                    """,
                    max_skills=max_skills,
                    search_term=search_term
                )
            else:
                top_skills_result = session.run(
                    """
                    MATCH (c:Candidate)-[]->(s)
                    WHERE NOT 'Candidate' IN labels(s)
                    WITH s, count(c) AS degree
                    ORDER BY degree DESC
                    LIMIT $max_skills
                    RETURN coalesce(s.name, s.id) AS name, degree
                    """,
                    max_skills=max_skills,
                )
                
            top_skills = top_skills_result.data()
            if not top_skills:
                return {"nodes": [], "edges": [], "stats": {}}

            skill_names = [r["name"] for r in top_skills if r["name"]]

            candidates_result = session.run(
                """
                MATCH (c:Candidate)-[]->(s)
                WHERE coalesce(s.name, s.id) IN $skill_names
                WITH c, count(s) AS degree
                ORDER BY degree DESC
                LIMIT $max_candidates
                RETURN c.id AS id, coalesce(c.name, c.canonical_id, c.id) AS name, degree
                """,
                skill_names=skill_names,
                max_candidates=max_candidates,
            )
            candidates = candidates_result.data()
            candidate_ids = [r["id"] for r in candidates]

            edges_result = session.run(
                """
                MATCH (c:Candidate)-[r]->(s)
                WHERE c.id IN $candidate_ids AND coalesce(s.name, s.id) IN $skill_names
                RETURN c.id AS source, coalesce(s.name, s.id) AS target, type(r) AS relation
                """,
                candidate_ids=candidate_ids,
                skill_names=skill_names,
            )
            edges = edges_result.data()

            stats_result = session.run(
                "MATCH (n) WITH coalesce(labels(n)[0], 'Entity') AS type, count(n) AS cnt RETURN type, cnt"
            )
            type_counts = {r["type"]: r["cnt"] for r in stats_result.data()}
            total_edges = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]

        nodes = []
        for s in top_skills:
            if not s["name"]: continue
            nodes.append({"id": s["name"], "label": str(s["name"]).title(), "type": "skill", "degree": s["degree"]})
        for c in candidates:
            label = (c["name"] or f"ID:{c['id']}")[:22]
            nodes.append({"id": c["id"], "label": label, "type": "candidate", "degree": c["degree"]})

        return {
            "nodes": nodes,
            "edges": [{"source": e["source"], "target": e["target"], "label": e["relation"].replace("HAS_", "").replace("_", " ").title()} for e in edges],
            "stats": {
                "total_nodes": sum(type_counts.values()),
                "total_edges": total_edges,
                "candidate_nodes": type_counts.get("Candidate", 0),
                "skill_nodes": sum(cnt for t, cnt in type_counts.items() if t != "Candidate"),
                "shown_nodes": len(nodes),
                "shown_edges": len(edges),
            },
        }
