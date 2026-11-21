from datetime import datetime, timezone

import pytest

from schotten.cards import FULL_DECK
from schotten.game import Game
from schotten.schemas import CreateGameRequest, Play
from schotten.neural.encoding import INPUT_SIZE, ACTION_SIZE, encode_view, legal_mask, action_id, decode_action


def position():
    game = Game("neural-test", CreateGameRequest(seed=7), datetime.now(timezone.utc))
    game.join(datetime.now(timezone.utc))
    return game.view(0)


def test_exact_encoding_preserves_cards_ties_and_stone_position():
    view = position()
    encoded = encode_view(view)
    assert INPUT_SIZE == len(encoded) == 1082
    assert sum(encoded[:54]) == 6
    assert sum(encoded[54:1026]) == 0
    assert encoded[1026:1029] == [1, 0, 0]
    assert encoded[1053:1056] == [1, 0, 0]
    assert encoded[-2:] == [1, 1]
    view.stones[4].yours = [FULL_DECK[8]]
    view.stones[4].theirs = [FULL_DECK[9]]
    view.stones[4].first_to_complete = "them"
    view.stones[4].claimed_by = "you"
    encoded = encode_view(view)
    assert encoded[54 + 4 * 108 + 8] == 1
    assert encoded[54 + 4 * 108 + 54 + 9] == 1
    assert encoded[1026 + 4 * 3:1026 + 5 * 3] == [0, 1, 0]
    assert encoded[1053 + 4 * 3:1053 + 5 * 3] == [0, 0, 1]


def test_encoding_is_order_independent_and_ignores_hidden_metadata():
    view = position()
    baseline = encode_view(view)
    view.hand.reverse()
    view.unseen_cards.clear()
    view.game_id = "different"
    view.version += 10
    assert encode_view(view) == baseline
    view.stones.reverse()
    assert encode_view(view) == baseline


def test_action_mapping_and_legal_mask_cover_every_placement():
    view = position()
    mask = legal_mask(view)
    assert ACTION_SIZE == len(mask) == 487
    assert sum(mask) == 54 and not mask[486]
    for card_id, card in enumerate(FULL_DECK):
        for stone in range(9):
            play = Play(card=card, stone=stone)
            assert action_id(play) == card_id * 9 + stone
            assert decode_action(action_id(play)) == play
            assert mask[action_id(play)] == (card in view.hand)
    assert decode_action(486) is None and action_id(None) == 486
    view.stones[0].claimed_by = "them"
    assert sum(legal_mask(view)) == 48
    view.hand = []
    assert legal_mask(view) == [False] * 486 + [True]
    view.phase = "game_over"
    assert not any(legal_mask(view))
    for invalid in [-1, 487, 1.5, True]:
        with pytest.raises(ValueError):
            decode_action(invalid)


def test_model_masks_illegal_moves_and_checkpoint_roundtrips(tmp_path):
    torch = pytest.importorskip("torch")
    from schotten.neural.model import Policy, masked_logits, save_checkpoint, load_checkpoint
    torch.manual_seed(1)
    model = Policy()
    assert sum(parameter.numel() for parameter in model.parameters()) == 468199
    state = torch.tensor([encode_view(position())])
    logits = model(state)
    mask = torch.tensor([legal_mask(position())])
    masked = masked_logits(logits, mask)
    assert mask[0, masked.argmax(dim=-1)[0]]
    assert torch.isneginf(masked[~mask]).all()
    with pytest.raises(ValueError):
        masked_logits(logits, torch.zeros_like(mask))
    path = tmp_path / "mlp.pt"
    save_checkpoint(path, model, epoch=2, metadata={"purpose":"test"})
    restored, payload = load_checkpoint(path)
    assert payload["epoch"] == 2
    assert payload["metadata"]["purpose"] == "test"
    assert torch.equal(restored(state), logits)
    payload["encoding_version"] = 99
    torch.save(payload, path)
    with pytest.raises(ValueError, match="encoding"):
        load_checkpoint(path)
