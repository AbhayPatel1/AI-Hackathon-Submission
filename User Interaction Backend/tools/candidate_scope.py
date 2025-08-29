from typing import List, Optional
from tools.name_matcher import match_names_to_emails

def extract_names_from_query(query: str) -> List[str]:
    # Basic: split query into words, return ones longer than 3
    words = query.lower().split()
    blacklist = {"tell", "what", "list", "about", "compare", "and", "who", "did", "the", "give"}
    filtered = [w for w in words if w not in blacklist and len(w) >= 3]
    
    
    return filtered

from typing import List, Optional, Dict
from tools.name_matcher import match_names_to_emails

def extract_names_from_query(query: str) -> List[str]:
    words = query.lower().split()
    blacklist = {"tell", "what", "list", "about", "compare", "and", "who", "did", "the", "give", "are"}
    return [w for w in words if w not in blacklist and len(w) >= 3]

def resolve_candidate_scope(query: str, selected_emails: Optional[List[str]] = None) -> List[Dict[str, str]]:
    query_names = extract_names_from_query(query)
    fuzzy_matches = match_names_to_emails(query_names) if query_names else []

    if selected_emails:
        # If emails were provided directly, try to enrich them with fuzzy name match info
        email_map = {m["email"]: m for m in fuzzy_matches}
        return [
            {
                "query_name": email_map.get(email, {}).get("query_name"),
                "matched_name": email_map.get(email, {}).get("matched_name"),
                "email": email
            }
            for email in selected_emails
        ]
    else:
        return fuzzy_matches