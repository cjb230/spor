"""Keep openapi.yaml (hand-written contract) and the FastAPI app from drifting apart."""
import re
from pathlib import Path

import yaml

from schotten.app import create_app
from schotten.errors import ERROR_CODES

SPEC = yaml.safe_load((Path(__file__).parent.parent / "openapi.yaml").read_text())
LIVE = create_app().openapi()
METHODS = {"get", "post", "put", "patch", "delete"}


def operations(paths):
    # path parameter names differ in style (gameId vs game_id); compare shape only
    return {
        (re.sub(r"\{[^}]+\}", "{}", path), method)
        for path, item in paths.items()
        for method in item
        if method in METHODS
    }


def test_routes_match_the_spec():
    assert operations(LIVE["paths"]) == operations(SPEC["paths"])


def test_error_codes_match_the_spec():
    spec_codes = SPEC["components"]["schemas"]["Error"]["properties"]["error"]["enum"]
    assert set(spec_codes) == set(ERROR_CODES)


def test_schema_fields_match_the_spec():
    spec, live = SPEC["components"]["schemas"], LIVE["components"]["schemas"]
    shared = set(spec) & set(live)
    must_compare = {"GameView", "TurnRequest", "LegalMoves", "CreateGameRequest", "SeatGrant",
                    "Stone", "Card", "LogEntry", "GameLog", "Play"}
    assert must_compare <= shared
    for name in must_compare:
        assert set(spec[name]["properties"]) == set(live[name]["properties"]), name
