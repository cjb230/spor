"""Versioned player-observation and action representation shared by training and play."""
from ..cards import FULL_DECK, sort_key
from ..schemas import GameView, Play

ENCODING_VERSION = 1
INPUT_SIZE = 1082
ACTION_SIZE = 487
PASS_ACTION = 486
SIDES = (None, "you", "them")


def encode_view(view: GameView) -> list[float]:
    if len(view.stones) != 9 or {stone.index for stone in view.stones} != set(range(9)):
        raise ValueError("an observation must contain all nine distinct stones")
    features = [0.0] * INPUT_SIZE
    for card in view.hand:
        features[sort_key(card)] = 1.0
    for stone in view.stones:
        for side_index, cards in enumerate((stone.yours, stone.theirs)):
            for card in cards:
                features[54 + stone.index * 108 + side_index * 54 + sort_key(card)] = 1.0
        features[1026 + stone.index * 3 + SIDES.index(stone.claimed_by)] = 1.0
        features[1053 + stone.index * 3 + SIDES.index(stone.first_to_complete)] = 1.0
    features[-2] = view.deck_count / 42.0
    features[-1] = view.opponent_hand_count / 6.0
    return features


def action_id(play: Play | None) -> int:
    return PASS_ACTION if play is None else sort_key(play.card) * 9 + play.stone


def decode_action(action: int) -> Play | None:
    if type(action) is not int or not 0 <= action < ACTION_SIZE:
        raise ValueError("action must be an integer from 0 to 486")
    return None if action == PASS_ACTION else Play(card=FULL_DECK[action // 9], stone=action % 9)


def legal_mask(view: GameView) -> list[bool]:
    mask = [False] * ACTION_SIZE
    if view.phase != "your_turn":
        return mask
    for stone in view.stones:
        if stone.claimed_by is None and len(stone.yours) < 3:
            for card in view.hand:
                mask[sort_key(card) * 9 + stone.index] = True
    mask[PASS_ACTION] = not any(mask)
    return mask
