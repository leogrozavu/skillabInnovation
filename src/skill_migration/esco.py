import ast
import re

import pandas as pd


ISCO_RE = re.compile(r"/isco/(C\d+)")


SECTOR_RULES = [
    (("C25",), "ict_software"),
    (("C21", "C31", "C35"), "engineering_science"),
    (("C22", "C32", "C53"), "healthcare_social_care"),
    (("C23",), "education"),
    (("C24", "C33", "C41", "C42", "C43"), "business_finance_admin"),
    (("C12", "C13", "C14"), "management_operations"),
    (("C26",), "arts_media_culture"),
    (("C51", "C52",), "retail_sales_hospitality"),
    (("C54",), "security_public_services"),
    (("C61", "C62", "C63"), "agriculture_forestry_fishery"),
    (("C71", "C72", "C73", "C74", "C75"), "trades_construction_craft"),
    (("C81", "C82",), "manufacturing_operations"),
    (("C83",), "transport_logistics"),
    (("C91", "C92", "C93", "C94", "C95", "C96"), "elementary_services"),
    (("C0",), "military"),
]


def safe_literal_eval(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def load_skill_mapping(path):
    columns = [
        "conceptUri",
        "preferredLabel",
        "altLabels",
        "description",
        "skills_ancestors",
        "traversal_ancestors",
        "children",
    ]
    df = pd.read_excel(path, usecols=columns)
    return df.rename(
        columns={
            "conceptUri": "skill_uri",
            "preferredLabel": "skill_label",
            "altLabels": "skill_alt_labels",
            "description": "skill_description",
        }
    )


def load_occupation_mapping(path):
    columns = ["conceptUri", "preferredLabel", "altLabels", "description", "ancestors", "levels"]
    df = pd.read_excel(path, usecols=columns)
    df = df.rename(
        columns={
            "conceptUri": "occupation_uri",
            "preferredLabel": "occupation_label",
            "altLabels": "occupation_alt_labels",
            "description": "occupation_description",
        }
    )
    df["isco_code"] = df["ancestors"].apply(extract_most_specific_isco)
    df["sector_proxy"] = df["isco_code"].apply(map_isco_to_sector)
    return df


def extract_most_specific_isco(ancestors):
    values = safe_literal_eval(ancestors)
    flattened = []

    def flatten(item):
        if isinstance(item, list):
            for child in item:
                flatten(child)
        else:
            flattened.append(str(item))

    flatten(values)
    codes = []
    for value in flattened:
        match = ISCO_RE.search(value)
        if match:
            codes.append(match.group(1))

    if not codes:
        return None

    return max(codes, key=len)


def map_isco_to_sector(isco_code):
    if not isinstance(isco_code, str) or not isco_code:
        return "unknown"

    for prefixes, sector in SECTOR_RULES:
        if any(isco_code.startswith(prefix) for prefix in prefixes):
            return sector

    broad = isco_code[:2]
    broad_map = {
        "C1": "management_operations",
        "C2": "professionals",
        "C3": "technicians_associate_professionals",
        "C4": "clerical_support",
        "C5": "services_sales",
        "C6": "agriculture_forestry_fishery",
        "C7": "trades_construction_craft",
        "C8": "plant_machine_operations",
        "C9": "elementary_services",
    }
    return broad_map.get(broad, "unknown")
