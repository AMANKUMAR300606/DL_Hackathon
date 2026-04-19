from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import uuid
import pandas as pd

from src.core.data_loader import load_and_clean_data
from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.faiss_indexer import FAISSIndexer
from src.indexing.graph_indexer import GraphIndexer
from src.core.retriever import HybridRetriever
from src.models.schema import CandidateProfile, Attribute

app = FastAPI(title="Explainable Hybrid Retrieval System")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

retriever: HybridRetriever | None = None
graph_indexer: GraphIndexer | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_data_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    candidates = [
        os.path.join(base_dir, "data", "profiles.csv"),
        os.path.join(base_dir, "..", "profiles.csv"),
        os.path.join(base_dir, "profiles.csv"),
    ]
    data_path = next((c for c in candidates if os.path.exists(c)), None)
    if data_path is None:
        raise FileNotFoundError("profiles.csv not found. Searched:\n" + "\n".join(candidates))
    return data_path


def _build_indexes():
    """Full rebuild from CSV. Called at startup and by POST /reload."""
    global retriever, graph_indexer

    data_path = _resolve_data_path()
    print(f"Loading profiles from: {data_path}")
    profiles = load_and_clean_data(data_path)
    print(f"  -> {len(profiles)} profiles loaded.")

    print("Initializing BM25 indexer...")
    bm25 = BM25Indexer()
    bm25.index(profiles)

    print("Initializing FAISS indexer...")
    faiss_idx = FAISSIndexer()
    faiss_idx.index(profiles)

    print("Initializing Neo4j graph indexer...")
    if graph_indexer is not None:
        try:
            graph_indexer.close()
        except Exception:
            pass
    graph_indexer = GraphIndexer()
    graph_indexer.index(profiles)

    retriever = HybridRetriever(bm25, faiss_idx, graph_indexer, profiles=profiles)
    print("System ready.")


COLUMN_GROUPS = {
    "core_skills":      "Skill",
    "secondary_skills": "Skill",
    "soft_skills":      "Soft Skill",
    "potential_roles":  "Role",
}
IGNORE_COLUMNS = {"id", "name", "years_of_experience", "skill_summary"}


def _extract_items(text: str):
    if not text:
        return []
    return [x.split("(")[0].strip() for x in str(text).split(",") if x.strip()]


def _build_profile(data: dict) -> CandidateProfile:
    """Build a CandidateProfile from a flat form/JSON dict."""
    pid = str(data.get("id", "")).strip() or str(uuid.uuid4())[:8]

    attributes = []
    for col, attr_type in COLUMN_GROUPS.items():
        raw = str(data.get(col, "") or "").strip()
        if not raw:
            continue
        items = _extract_items(raw)
        if items:
            attributes.append(Attribute(key=col, value=items, type=attr_type))

    try:
        yoe = float(data.get("years_of_experience", 0) or 0)
    except (ValueError, TypeError):
        yoe = 0.0

    return CandidateProfile(
        id=pid,
        name=str(data.get("name", "") or "").strip() or None,
        core_skills=str(data.get("core_skills", "") or "").strip() or None,
        secondary_skills=str(data.get("secondary_skills", "") or "").strip() or None,
        soft_skills=str(data.get("soft_skills", "") or "").strip() or None,
        years_of_experience=yoe,
        potential_roles=str(data.get("potential_roles", "") or "").strip() or None,
        skill_summary=str(data.get("skill_summary", "") or "").strip() or None,
        attributes=attributes,
        entity_type="candidate",
    )


def _csv_id_exists(pid: str) -> bool:
    try:
        df = pd.read_csv(_resolve_data_path(), usecols=["id"], dtype=str)
        return pid in df["id"].values
    except Exception:
        return False


def _append_csv(profile: CandidateProfile):
    """Append new profile row to CSV."""
    row = {
        "id": profile.id,
        "name": profile.name or "",
        "core_skills": profile.core_skills or "",
        "secondary_skills": profile.secondary_skills or "",
        "soft_skills": profile.soft_skills or "",
        "years_of_experience": profile.years_of_experience,
        "potential_roles": profile.potential_roles or "",
        "skill_summary": profile.skill_summary or "",
    }
    path = _resolve_data_path()
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _update_csv(profile: CandidateProfile):
    """Update or insert a profile row in CSV."""
    path = _resolve_data_path()
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        df = pd.DataFrame()

    row = {
        "id": str(profile.id),
        "name": profile.name or "",
        "core_skills": profile.core_skills or "",
        "secondary_skills": profile.secondary_skills or "",
        "soft_skills": profile.soft_skills or "",
        "years_of_experience": str(profile.years_of_experience),
        "potential_roles": profile.potential_roles or "",
        "skill_summary": profile.skill_summary or "",
    }

    mask = df["id"].astype(str) == str(profile.id) if "id" in df.columns else pd.Series([False] * len(df))
    if mask.any():
        for col, val in row.items():
            if col in df.columns:
                df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    _build_indexes()


@app.on_event("shutdown")
async def shutdown_event():
    if graph_indexer:
        graph_indexer.close()


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request, "results": None, "query": ""})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...), limit: str = Form("10")):
    top_k = 99999 if limit == "max" else int(limit)
    results = retriever.search(query, top_k=top_k)
    intent_info = retriever.intent_detector.analyze_intent(query)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "results": results, "query": query, "intent": intent_info, "limit": limit},
    )


@app.get("/graph", response_class=HTMLResponse)
async def graph_page(request: Request):
    return templates.TemplateResponse(request=request, name="graph.html", context={"request": request})


@app.get("/graph-data")
async def graph_data(max_skills: int = 50, max_candidates: int = 60, search_term: str = None):
    """Return graph nodes/edges from Neo4j for D3.js visualization."""
    data = graph_indexer.get_graph_data(max_skills=max_skills, max_candidates=max_candidates, search_term=search_term)
    return JSONResponse(data)


# ─────────────────────────────────────────────────────────────────────────────
# Profile Management API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/profile/{profile_id}")
async def get_profile(profile_id: str):
    """
    Return a single profile's data as JSON for pre-filling the update form.
    Searches the in-memory index (fast, no CSV read).
    """
    profile = graph_indexer._profile_map.get(profile_id)
    if profile is None:
        # Try searching by name (case-insensitive) as fallback
        lower = profile_id.lower()
        for p in retriever.bm25.profiles:
            if (p.name or "").lower() == lower or p.id == profile_id:
                profile = p
                break
    if profile is None:
        return JSONResponse({"error": f"Profile '{profile_id}' not found."}, status_code=404)

    return JSONResponse({
        "id":                 profile.id,
        "name":               profile.name or "",
        "core_skills":        profile.core_skills or "",
        "secondary_skills":   profile.secondary_skills or "",
        "soft_skills":        profile.soft_skills or "",
        "years_of_experience": profile.years_of_experience,
        "potential_roles":    profile.potential_roles or "",
        "skill_summary":      profile.skill_summary or "",
    })


@app.post("/profile/new")
async def add_profile_form(request: Request):
    """
    Add a brand-new profile submitted from the frontend form.
    Builds CandidateProfile from flat JSON, saves to CSV, and updates all
    in-memory indexes incrementally (no full rebuild).
    """
    import json as _json
    try:
        data = _json.loads(await request.body())
        profile = _build_profile(data)

        if _csv_id_exists(profile.id):
            return JSONResponse({"status": "error", "message": f"ID {profile.id} already exists."}, status_code=409)

        _append_csv(profile)
        retriever.bm25.add_single(profile)
        retriever.faiss.add_single(profile)
        graph_indexer.index_single(profile)
        retriever.intent_detector.extend([profile])

        print(f"[/profile/new] Added profile {profile.id} ({profile.name}) to all indexes.")
        return JSONResponse({"status": "ok", "id": profile.id, "message": f"Profile '{profile.name or profile.id}' added successfully."})

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.put("/profile/update")
async def update_profile_form(request: Request):
    """
    Update an existing profile submitted from the frontend form.
    Updates CSV row + all in-memory indexes incrementally.
    """
    import json as _json
    try:
        data = _json.loads(await request.body())
        if not str(data.get("id", "")).strip():
            return JSONResponse({"status": "error", "message": "Profile ID is required for update."}, status_code=400)

        profile = _build_profile(data)
        _update_csv(profile)
        retriever.bm25.update_single(profile)
        retriever.faiss.update_single(profile)
        graph_indexer.update_single(profile)
        retriever.intent_detector.extend([profile])

        print(f"[/profile/update] Updated profile {profile.id} ({profile.name}) in all indexes.")
        return JSONResponse({"status": "ok", "id": profile.id, "message": f"Profile '{profile.name or profile.id}' updated successfully."})

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/add-profile")
async def add_profile_endpoint(request: Request):
    """
    Incrementally add ONE new profile to all in-memory indexes.
    Called automatically by input.py after writing to CSV.
    """
    import json as _json
    try:
        body = _json.loads(await request.body())
        profile = CandidateProfile(**body)
        retriever.bm25.add_single(profile)
        retriever.faiss.add_single(profile)
        graph_indexer.index_single(profile)
        retriever.intent_detector.extend([profile])
        print(f"[/add-profile] Incrementally added profile {profile.id} to all indexes.")
        return JSONResponse({"status": "ok", "message": f"Profile {profile.id} added to all indexes."})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/reload")
async def reload_indexes():
    """Full rebuild of all indexes from the current profiles.csv."""
    try:
        _build_indexes()
        return JSONResponse({"status": "ok", "message": "Indexes reloaded successfully."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
