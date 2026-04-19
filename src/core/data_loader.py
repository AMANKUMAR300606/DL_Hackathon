import pandas as pd
from typing import List
import os

from src.models.schema import CandidateProfile, Attribute

def load_and_clean_data(csv_path: str) -> List[CandidateProfile]:
    """
    Loads profiles.csv, removes nulls and duplicates,
    and returns a list of CandidateProfile Pydantic models.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Drop duplicates based on id
    df = df.drop_duplicates(subset=['id'], keep='first')

    # We will cast id to string for consistency
    df['id'] = df['id'].astype(str)

    # Replace pandas NA with None or empty string for processing
    df = df.fillna("")

    profiles = []
    for _, row in df.iterrows():
        try:
            # Parse year of experience carefully
            yoe = row.get('years_of_experience', 0.0)
            if pd.isna(yoe) or yoe == "":
                yoe = 0.0
            else:
                yoe = float(yoe)

            def extract_items(text):
                if pd.isna(text) or text == "": return []
                return [x.split("(")[0].strip() for x in str(text).split(",") if x.strip()]

            # 1. Maintain perfectly explicit Legacy fields so BM25 & FAISS aren't mathematically lobotomized
            legacy_core = row.get('core_skills', '')
            legacy_sec = row.get('secondary_skills', '')
            legacy_soft = row.get('soft_skills', '')
            legacy_roles = row.get('potential_roles', '')
            legacy_summary = row.get('skill_summary', '')

            # 2. Universal Schema Loader for Graph Nodes
            IGNORE_COLUMNS = {"id", "name", "years_of_experience", "skill_summary"}
            COLUMN_GROUPS = {
                "core_skills": "Skill",
                "secondary_skills": "Skill",
                "soft_skills": "Soft Skill",
                "potential_roles": "Role"
            }
            attributes = []

            for col in df.columns:
                if col in IGNORE_COLUMNS: continue
                val = row.get(col)
                if pd.isna(val) or val == "": continue
                
                items = extract_items(val)
                if not items: continue
                
                str_col = str(col)
                # Make header gracefully readable (e.g. "core_skills" -> "Skill")
                clean_key = COLUMN_GROUPS.get(str_col, str_col.replace("_", " ").title())
                attributes.append(Attribute(key=clean_key, value=items))

            profile = CandidateProfile(
                id=str(row['id']),
                name=row['name'] if pd.notna(row.get('name')) and row.get('name') != "" else None,
                core_skills=legacy_core if pd.notna(legacy_core) and legacy_core != "" else None,
                secondary_skills=legacy_sec if pd.notna(legacy_sec) and legacy_sec != "" else None,
                soft_skills=legacy_soft if pd.notna(legacy_soft) and legacy_soft != "" else None,
                years_of_experience=yoe,
                potential_roles=legacy_roles if pd.notna(legacy_roles) and legacy_roles != "" else None,
                skill_summary=legacy_summary if pd.notna(legacy_summary) and legacy_summary != "" else None,
                attributes=attributes
            )
            profiles.append(profile)
        except Exception as e:
            # Skip rows that fail validation
            print(f"Skipping row {row['id']} due to error: {e}")
            continue

    return profiles

if __name__ == "__main__":
    profiles = load_and_clean_data("../data/profiles.csv")
    print(f"Loaded {len(profiles)} cleaned profiles.")
