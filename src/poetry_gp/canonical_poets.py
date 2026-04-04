"""
Curated list of canonical poets for prioritization in visualizations.

This list represents poets of historical and literary significance who should
be highlighted in visualizations when present in the corpus, regardless of
poem count.
"""

from __future__ import annotations

# Canonical poets across eras and traditions
# Organized roughly by era for clarity
CANONICAL_POETS = {
    # Medieval & Renaissance
    "geoffrey chaucer",
    "william shakespeare",
    "john donne",
    "ben jonson",
    "john milton",
    "andrew marvell",

    # 18th Century
    "alexander pope",
    "jonathan swift",
    "william blake",

    # Romantic Era
    "william wordsworth",
    "samuel taylor coleridge",
    "lord byron",
    "percy bysshe shelley",
    "john keats",

    # Victorian Era
    "alfred tennyson",
    "robert browning",
    "elizabeth barrett browning",
    "matthew arnold",
    "gerard manley hopkins",
    "thomas hardy",

    # American 19th Century
    "walt whitman",
    "emily dickinson",
    "edgar allan poe",
    "henry wadsworth longfellow",

    # Modernist Era
    "william butler yeats",
    "t s eliot",
    "ezra pound",
    "robert frost",
    "wallace stevens",
    "marianne moore",
    "william carlos williams",
    "e e cummings",
    "hart crane",

    # British Modern & Mid-20th Century
    "w h auden",
    "dylan thomas",
    "stephen spender",
    "philip larkin",
    "ted hughes",
    "sylvia plath",
    "seamus heaney",

    # American Mid-20th Century
    "robert lowell",
    "elizabeth bishop",
    "john berryman",
    "robert hayden",
    "gwendolyn brooks",
    "langston hughes",
    "allen ginsberg",

    # Contemporary (representative selection)
    "maya angelou",
    "adrienne rich",
    "derek walcott",
    "louise gluck",
    "mary oliver",
}


def normalize_poet_name(name: str) -> str:
    """Normalize poet name for comparison (lowercase, collapsed whitespace)."""
    return " ".join(str(name).strip().lower().split())


def is_canonical(poet_name: str) -> bool:
    """Check if a poet is in the canonical list."""
    normalized = normalize_poet_name(poet_name)
    return normalized in CANONICAL_POETS
