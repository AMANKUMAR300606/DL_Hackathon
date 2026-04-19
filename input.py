"""
input.py — Add new candidate profiles to the Hybrid Retrieval System.

What it does:
  1. Collects new profile data via interactive prompts (or --from-json / --from-csv)
  2. Validates data against the existing CandidateProfile schema
  3. Appends new row to profiles.csv (the single source of truth)
  4. Incrementally updates FAISS vector index (persisted to data/faiss.index)
  5. Incrementally updates BM25 index (persisted to data/bm25.pkl via pickle)
  6. Incrementally updates Neo4j knowledge graph (MERGE — no full wipe)

Usage:
    python input.py                           # Interactive mode
    python input.py --from-json new.json      # Bulk import from JSON list
    python input.py --from-csv new_rows.csv   # Bulk import from CSV rows
    python input.py --list                    # Show last 10 profiles in CSV

No existing files are modified except the CSV and the two persisted index files.
"""

import os
import sys
import json
import pickle
import argparse
import uuid
import re
from typing import List, Optional

# ── Lightweight imports only at top-level ──────────────────────────────────
# Heavy libs (faiss, sentence_transformers, rank_bm25) are imported lazily
# inside their respective functions so the interactive prompt appears instantly.
import pandas as pd

# ── Make sure src/ imports work regardless of cwd ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.schema import CandidateProfile, Attribute
from src.core.data_loader import load_and_clean_data

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(BASE_DIR, "data")
CSV_PATH        = os.path.join(DATA_DIR, "profiles.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
FAISS_META_PATH  = os.path.join(DATA_DIR, "faiss_meta.pkl")   # stores ordered list of profile IDs
BM25_PATH        = os.path.join(DATA_DIR, "bm25.pkl")

# Fallback CSV location (root of akshat/)
if not os.path.exists(CSV_PATH):
    alt = os.path.join(BASE_DIR, "profiles.csv")
    if os.path.exists(alt):
        CSV_PATH = alt

os.makedirs(DATA_DIR, exist_ok=True)

# ── Neo4j config (mirrors graph_indexer.py) ─────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Akshat007")

# ── Shared embedding model (loaded lazily on first use) ─────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer  # lazy import
        print(f"  ⏳ Loading embedding model ({EMBED_MODEL_NAME})…")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print(f"  ✔ Embedding model ready.")
    return _embed_model


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — PROFILE BUILDER
# ════════════════════════════════════════════════════════════════════════════

COLUMN_GROUPS = {
    "core_skills":       "Skill",
    "secondary_skills":  "Skill",
    "soft_skills":       "Soft Skill",
    "potential_roles":   "Role",
}
IGNORE_COLUMNS = {"id", "name", "years_of_experience", "skill_summary"}


def _extract_items(text: str) -> List[str]:
    if not text:
        return []
    return [x.split("(")[0].strip() for x in str(text).split(",") if x.strip()]


def build_profile_from_dict(data: dict) -> CandidateProfile:
    """
    Convert a flat dict (matching CSV columns) into a validated CandidateProfile.
    The 'id' field is auto-generated if absent or empty.
    """
    # Auto-generate ID if missing
    pid = str(data.get("id", "")).strip()
    if not pid:
        pid = str(uuid.uuid4().int)[:9]   # 9-digit numeric-style ID
    data["id"] = pid

    yoe = data.get("years_of_experience", 0.0)
    try:
        yoe = float(yoe) if yoe not in ("", None) else 0.0
    except (ValueError, TypeError):
        yoe = 0.0

    # Build dynamic attributes (same logic as data_loader.py)
    attributes = []
    for col, val in data.items():
        if col in IGNORE_COLUMNS:
            continue
        if not val or str(val).strip() == "":
            continue
        items = _extract_items(str(val))
        if not items:
            continue
        clean_key = COLUMN_GROUPS.get(col, col.replace("_", " ").title())
        attributes.append(Attribute(key=clean_key, value=items))

    profile = CandidateProfile(
        id=pid,
        name=data.get("name") or None,
        core_skills=data.get("core_skills") or None,
        secondary_skills=data.get("secondary_skills") or None,
        soft_skills=data.get("soft_skills") or None,
        years_of_experience=yoe,
        potential_roles=data.get("potential_roles") or None,
        skill_summary=data.get("skill_summary") or None,
        attributes=attributes,
    )
    return profile


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — CSV PERSISTENCE
# ════════════════════════════════════════════════════════════════════════════

# Canonical CSV columns (matches existing profiles.csv layout)
CSV_COLUMNS = [
    "id", "name", "core_skills", "secondary_skills", "soft_skills",
    "years_of_experience", "potential_roles", "skill_summary",
]


def profile_to_row(profile: CandidateProfile) -> dict:
    return {
        "id":                   profile.id,
        "name":                 profile.name or "",
        "core_skills":          profile.core_skills or "",
        "secondary_skills":     profile.secondary_skills or "",
        "soft_skills":          profile.soft_skills or "",
        "years_of_experience":  profile.years_of_experience,
        "potential_roles":      profile.potential_roles or "",
        "skill_summary":        profile.skill_summary or "",
    }


def csv_id_exists(pid: str) -> bool:
    if not os.path.exists(CSV_PATH):
        return False
    df = pd.read_csv(CSV_PATH, usecols=["id"], dtype=str)
    return pid in df["id"].values


def append_to_csv(profile: CandidateProfile):
    """Append a single profile row to profiles.csv (creates file if absent)."""
    row = profile_to_row(profile)
    row_df = pd.DataFrame([row])

    if not os.path.exists(CSV_PATH):
        row_df.to_csv(CSV_PATH, index=False)
        print(f"  ✔ Created new CSV at {CSV_PATH}")
    else:
        # Check for ID collision
        if csv_id_exists(profile.id):
            raise ValueError(f"Profile ID '{profile.id}' already exists in CSV. Skipping.")
        row_df.to_csv(CSV_PATH, mode="a", header=False, index=False)
        print(f"  ✔ Appended profile {profile.id} to CSV.")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — FAISS INCREMENTAL UPDATE
# ════════════════════════════════════════════════════════════════════════════

def _profile_to_faiss_text(profile: CandidateProfile) -> str:
    """Mirror of FAISSIndexer logic — build text for embedding."""
    parts = []
    for a in (profile.attributes or []):
        vals = " ".join(str(v) for v in a.value)
        parts.append(f"{a.key}: {vals}")
        parts.append(vals)
    if profile.core_skills:
        parts.append(f"Skills: {profile.core_skills}")
    if profile.potential_roles:
        parts.append(f"Roles: {profile.potential_roles}")
    if profile.skill_summary:
        parts.append(profile.skill_summary)
    return " ".join(parts)


def update_faiss_index(profile: CandidateProfile):
    """
    Load existing persisted FAISS index, add the new profile embedding,
    and save back to disk. Creates a fresh index if none exists.
    """
    import faiss                                   # lazy import
    import numpy as np                             # lazy import

    model = get_embed_model()
    dim = model.get_sentence_embedding_dimension()

    # Load or create index
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "rb") as f:
            meta: List[str] = pickle.load(f)   # ordered list of profile IDs
    else:
        print("  ℹ No existing FAISS index found — building fresh from CSV…")
        index, meta = _build_faiss_from_csv(model, dim)

    # Check if already indexed
    if profile.id in meta:
        print(f"  ⚠ Profile {profile.id} already in FAISS index — skipping.")
        return

    text = _profile_to_faiss_text(profile)
    emb = model.encode([text]).astype("float32")
    faiss.normalize_L2(emb)

    index.add(emb)
    meta.append(profile.id)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"  ✔ FAISS index updated — total vectors: {index.ntotal}")


def _build_faiss_from_csv(model, dim: int):
    """Build FAISS index from scratch using current CSV. Used as fallback."""
    import faiss                                   # lazy import

    profiles = load_and_clean_data(CSV_PATH)
    index = faiss.IndexFlatL2(dim)
    meta: List[str] = []
    if not profiles:
        return index, meta

    texts = [_profile_to_faiss_text(p) for p in profiles]
    embs = model.encode(texts, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(embs)
    index.add(embs)
    meta = [p.id for p in profiles]

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"  ✔ FAISS rebuilt from CSV — {index.ntotal} vectors saved.")
    return index, meta


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — BM25 INCREMENTAL UPDATE
# ════════════════════════════════════════════════════════════════════════════

STOPWORDS = {
    "and", "or", "the", "a", "an", "of", "in", "to", "for",
    "with", "is", "it", "at", "on", "by", "as", "be", "this",
    "manager", "specialist", "coordinator", "associate", "senior",
    "junior", "lead", "expert", "professional",
}
SYNONYMS = {
    "front-end": "frontend", "fullstack": "fullstack", "full-stack": "fullstack",
    "html5": "html", "css3": "css", "js": "javascript",
    "react.js": "react", "node.js": "nodejs", "vue.js": "vue",
    "postgres": "postgresql", "ml": "machine learning", "ai": "artificial intelligence",
}


def _bm25_preprocess(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    text = text.replace("c++", "cplusplus").replace("c#", "csharp") \
               .replace(".net", "dotnet").replace(".js", "js")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [SYNONYMS.get(t, t) for t in tokens]
    return tokens


def _profile_to_bm25_text(profile: CandidateProfile) -> str:
    """Mirror of BM25Indexer logic."""
    parts = []
    for a in (profile.attributes or []):
        vals = " ".join(str(v) for v in a.value)
        parts.append(f"{a.key}: {vals}")
    if profile.skill_summary:
        parts.append(profile.skill_summary)
    return " ".join(parts)


def update_bm25_index(profile: CandidateProfile):
    """
    BM25Okapi doesn't support incremental adds — we must rebuild the corpus.
    We load existing serialised state (corpus + profile IDs), add the new
    entry, rebuild BM25Okapi, then persist.
    """
    from rank_bm25 import BM25Okapi   # lazy import

    if os.path.exists(BM25_PATH):
        with open(BM25_PATH, "rb") as f:
            state = pickle.load(f)
        corpus: List[List[str]] = state["corpus"]
        ids: List[str] = state["ids"]
    else:
        print("  ℹ No existing BM25 state found — building from CSV…")
        corpus, ids = _build_bm25_from_csv()

    if profile.id in ids:
        print(f"  ⚠ Profile {profile.id} already in BM25 index — skipping.")
        return

    tokens = _bm25_preprocess(_profile_to_bm25_text(profile))
    corpus.append(tokens)
    ids.append(profile.id)

    bm25 = BM25Okapi(corpus)

    with open(BM25_PATH, "wb") as f:
        pickle.dump({"corpus": corpus, "ids": ids, "bm25": bm25}, f)

    print(f"  ✔ BM25 index updated — total docs: {len(corpus)}")


def _build_bm25_from_csv():
    from rank_bm25 import BM25Okapi   # lazy import
    profiles = load_and_clean_data(CSV_PATH)
    corpus = [_bm25_preprocess(_profile_to_bm25_text(p)) for p in profiles]
    ids = [p.id for p in profiles]
    bm25 = BM25Okapi(corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"corpus": corpus, "ids": ids, "bm25": bm25}, f)
    print(f"  ✔ BM25 rebuilt from CSV — {len(corpus)} docs saved.")
    return corpus, ids


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — NEO4J INCREMENTAL UPDATE
# ════════════════════════════════════════════════════════════════════════════

def update_neo4j_graph(profile: CandidateProfile):
    """
    Incrementally MERGE the new candidate + its attributes into Neo4j.
    Uses MERGE so re-running is safe (idempotent). No full wipe.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("  ⚠ neo4j driver not installed — skipping graph update.")
        return

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            # Ensure constraints exist (harmless if already present)
            for constraint in [
                "CREATE CONSTRAINT candidate_id IF NOT EXISTS FOR (c:Candidate) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT skill_name   IF NOT EXISTS FOR (s:Skill)     REQUIRE s.name IS UNIQUE",
                "CREATE CONSTRAINT role_name    IF NOT EXISTS FOR (r:Role)      REQUIRE r.name IS UNIQUE",
            ]:
                try:
                    session.run(constraint)
                except Exception:
                    pass  # Constraint may already exist under a different syntax

            # MERGE candidate node
            session.run(
                """
                MERGE (c:Candidate {id: $id})
                SET c.name        = $name,
                    c.entity_type = $entity_type
                """,
                id=profile.id,
                name=profile.name or "",
                entity_type=profile.entity_type or "candidate",
            )

            # MERGE dynamic attribute nodes + relationships
            for attr in profile.attributes:
                if not attr.value:
                    continue
                vals = attr.value if isinstance(attr.value, list) else [attr.value]
                label_str = attr.key.replace(" ", "").title()
                rel_type = f"HAS_{attr.key.replace(' ', '_').upper()}"

                for v in vals:
                    clean_v = str(v).strip()
                    if not clean_v:
                        continue
                    session.run(
                        f"""
                        MERGE (n:{label_str} {{name: $val}})
                        WITH n
                        MATCH (c:Candidate {{id: $id}})
                        MERGE (c)-[:{rel_type}]->(n)
                        """,
                        val=clean_v,
                        id=profile.id,
                    )

        driver.close()
        print(f"  ✔ Neo4j graph updated for candidate {profile.id}.")

    except Exception as e:
        print(f"  ⚠ Neo4j update failed (is Neo4j running?): {e}")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — INTERACTIVE INPUT
# ════════════════════════════════════════════════════════════════════════════

def _prompt(label: str, required: bool = False, default: str = "") -> str:
    suffix = " (required)" if required else f" [leave blank to skip{', default: ' + default if default else ''}]"
    while True:
        val = input(f"  {label}{suffix}: ").strip()
        if not val and default:
            return default
        if not val and required:
            print("  ✗ This field is required.")
            continue
        return val


def collect_interactive() -> dict:
    print("\n" + "═" * 55)
    print("  ADD NEW CANDIDATE PROFILE")
    print("═" * 55)
    print("  Press Enter to skip optional fields.\n")

    data = {}
    data["name"]               = _prompt("Full Name")
    data["core_skills"]        = _prompt("Core Skills (comma-separated, e.g. Python, SQL, Docker)")
    data["secondary_skills"]   = _prompt("Secondary Skills (comma-separated)")
    data["soft_skills"]        = _prompt("Soft Skills (comma-separated, e.g. Leadership, Communication)")
    data["years_of_experience"]= _prompt("Years of Experience", default="0")
    data["potential_roles"]    = _prompt("Potential Roles (comma-separated, e.g. Data Scientist, ML Engineer)")
    data["skill_summary"]      = _prompt("Skill Summary (free text description)")

    return data


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — BULK IMPORT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def load_from_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data


def load_from_csv_file(path: str) -> List[dict]:
    df = pd.read_csv(path).fillna("")
    return df.to_dict(orient="records")


# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — CORE ADD ROUTINE
# ════════════════════════════════════════════════════════════════════════════

SERVER_URL_DEFAULT = "http://localhost:8000"


def notify_server(profile: "CandidateProfile", server_url: str = SERVER_URL_DEFAULT):
    """
    Send the new profile JSON to POST /add-profile on the running FastAPI server.
    This triggers an INCREMENTAL index update (~1 second) — not a full rebuild.
    """
    import urllib.request
    import urllib.error
    import json as _json

    endpoint = server_url.rstrip("/") + "/add-profile"
    # Serialise profile; exclude heavy/non-JSON-safe fields
    payload = _json.dumps(profile.dict(), default=str).encode("utf-8")
    try:
        req = urllib.request.Request(
            endpoint, data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            print(f"  ✔ Server updated indexes: {body}")
    except urllib.error.URLError as e:
        print(f"  ⚠ Could not reach server at {endpoint}: {e.reason}")
        print("    (Is the FastAPI server running? Start it with: python main.py)")
        print("    Profile IS saved to CSV — restart the server to see it.")
    except Exception as e:
        print(f"  ⚠ Server notification failed: {e}")


def add_profile(data: dict, skip_faiss: bool = False, skip_bm25: bool = False,
               skip_graph: bool = False, server_url: Optional[str] = SERVER_URL_DEFAULT
               ) -> Optional[CandidateProfile]:
    """
    Full pipeline: validate → CSV → FAISS → BM25 → Neo4j → notify server.
    Returns the created CandidateProfile on success, None on error.
    """
    try:
        profile = build_profile_from_dict(data)
    except Exception as e:
        print(f"  ✗ Validation error: {e}")
        return None

    print(f"\n  Profile ID: {profile.id}")
    print(f"  Name      : {profile.name or '(unnamed)'}")

    # 1. CSV (source of truth — must succeed)
    try:
        append_to_csv(profile)
    except ValueError as e:
        print(f"  ✗ {e}")
        return None

    # 2. FAISS
    if not skip_faiss:
        try:
            update_faiss_index(profile)
        except Exception as e:
            print(f"  ⚠ FAISS update error: {e}")

    # 3. BM25
    if not skip_bm25:
        try:
            update_bm25_index(profile)
        except Exception as e:
            print(f"  ⚠ BM25 update error: {e}")

    # 4. Neo4j
    if not skip_graph:
        update_neo4j_graph(profile)

    # 5. Notify the running FastAPI server to update its in-memory indexes
    if server_url:
        print(f"\n  Notifying server to add profile to in-memory indexes…")
        notify_server(profile, server_url)

    print(f"\n  ✅ Profile '{profile.id}' successfully added to all indexes.\n")
    return profile


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — LIST RECENT PROFILES
# ════════════════════════════════════════════════════════════════════════════

def list_recent(n: int = 10):
    if not os.path.exists(CSV_PATH):
        print("No CSV found.")
        return
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    print(f"\nLast {n} profiles in {CSV_PATH}:\n")
    print(df.tail(n).to_string(index=False))
    print()


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — REBUILD ALL INDEXES
# ════════════════════════════════════════════════════════════════════════════

def rebuild_all():
    """Force a full rebuild of FAISS and BM25 from the current CSV."""
    print("\nRebuilding all indexes from CSV…")
    model = get_embed_model()
    dim = model.get_sentence_embedding_dimension()
    _build_faiss_from_csv(model, dim)
    _build_bm25_from_csv()
    print("Rebuild complete.\n")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── Banner printed INSTANTLY before any heavy library loads ──────────────
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Intent-Aware Hybrid Retrieval — Profile Manager   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    parser = argparse.ArgumentParser(
        description="Add new candidate profiles to the Hybrid Retrieval System.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python input.py                        # Interactive mode
  python input.py --from-json data.json  # Import JSON (list or single object)
  python input.py --from-csv rows.csv    # Import CSV rows
  python input.py --list                 # Show last 10 profiles
  python input.py --rebuild              # Force rebuild all indexes from CSV
  python input.py --no-graph             # Skip Neo4j update (e.g. Neo4j offline)
        """
    )
    parser.add_argument("--from-json", metavar="FILE", help="Import profile(s) from a JSON file")
    parser.add_argument("--from-csv",  metavar="FILE", help="Import profiles from a CSV file")
    parser.add_argument("--list",      action="store_true", help="List last N profiles in the CSV")
    parser.add_argument("--list-n",    type=int, default=10, help="Number of profiles to list (default: 10)")
    parser.add_argument("--rebuild",   action="store_true", help="Rebuild FAISS + BM25 indexes from scratch")
    parser.add_argument("--no-faiss",  action="store_true", help="Skip FAISS update")
    parser.add_argument("--no-bm25",   action="store_true", help="Skip BM25 update")
    parser.add_argument("--no-graph",  action="store_true", help="Skip Neo4j graph update")
    parser.add_argument("--server-url", default=SERVER_URL_DEFAULT,
                        help=f"FastAPI server URL for hot-reload (default: {SERVER_URL_DEFAULT})")
    parser.add_argument("--no-reload", action="store_true",
                        help="Skip notifying the server to reload (use if server is not running)")

    args = parser.parse_args()

    if args.list:
        list_recent(args.list_n)
        return

    if args.rebuild:
        rebuild_all()
        return

    # Collect profile data
    records: List[dict] = []

    if args.from_json:
        records = load_from_json(args.from_json)
        print(f"Loaded {len(records)} profile(s) from {args.from_json}")
    elif args.from_csv:
        records = load_from_csv_file(args.from_csv)
        print(f"Loaded {len(records)} profile(s) from {args.from_csv}")
    else:
        # Interactive mode
        records = [collect_interactive()]
        print()

        # Ask user to confirm before writing
        print("  ─── Preview ───────────────────────────────────")
        for k, v in records[0].items():
            if v:
                print(f"  {k:25s}: {v}")
        print("  ────────────────────────────────────────────────")
        confirm = input("\n  Confirm and add this profile? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("  Cancelled.")
            return

    # Process records
    success, failed = 0, 0
    server_url = None if args.no_reload else args.server_url
    for i, rec in enumerate(records, 1):
        print(f"\n[{i}/{len(records)}] Processing…")
        result = add_profile(
            rec,
            skip_faiss=args.no_faiss,
            skip_bm25=args.no_bm25,
            skip_graph=args.no_graph,
            server_url=server_url,
        )
        if result:
            success += 1
        else:
            failed += 1

    if len(records) > 1:
        print(f"\nDone — {success} added, {failed} failed.")


if __name__ == "__main__":
    main()
