"""Error codes and the exception the engine raises for rejected requests."""

# code -> HTTP status. Keep in sync with Error.error in openapi.yaml
# (tests/test_spec_parity.py enforces this).
ERROR_CODES = {
    "UNAUTHORIZED": 401,
    "NOT_FOUND": 404,
    "GAME_NOT_STARTED": 403,
    "GAME_OVER": 403,
    "NOT_YOUR_TURN": 403,
    "STALE_VERSION": 409,
    "GAME_FULL": 409,
    "CARD_NOT_IN_HAND": 422,
    "STONE_FULL": 422,
    "STONE_ALREADY_CLAIMED": 422,
    "CLAIM_INCOMPLETE": 422,
    "CLAIM_NOT_PROVABLE": 422,
    "PLAY_REQUIRED": 422,
    "INVALID_REQUEST": 422,
}


class GameError(Exception):
    def __init__(self, code: str, detail: str = ""):
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code {code}")
        self.code = code
        self.status = ERROR_CODES[code]
        self.detail = detail
        super().__init__(f"{code}: {detail}")
