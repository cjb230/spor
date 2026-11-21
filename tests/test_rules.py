from schotten.cards import Card, Color
from schotten.rules import check_win, claim_is_provable, evaluate, unseen_pool


def c(color, value):
    return Card(Color(color), value)


def test_formation_beats_total():
    color_run = evaluate([c("red", 1), c("red", 2), c("red", 3)])
    trips = evaluate([c("red", 5), c("blue", 5), c("green", 5)])
    color = evaluate([c("red", 1), c("red", 5), c("red", 9)])
    run = evaluate([c("red", 4), c("blue", 5), c("green", 6)])
    plain = evaluate([c("red", 9), c("blue", 8), c("green", 6)])  # total 23, but only a sum
    assert color_run > trips > color > run > plain


def test_run_is_order_independent():
    assert evaluate([c("red", 6), c("blue", 4), c("green", 5)]) == (1, 15)


def test_equal_formations_compare_by_total():
    low = evaluate([c("red", 1), c("blue", 2), c("green", 3)])
    high = evaluate([c("red", 7), c("blue", 8), c("green", 9)])
    assert high > low


def test_complete_stone_compares_directly_with_tiebreak():
    mine = [c("red", 1), c("blue", 2), c("green", 3)]
    theirs = [c("yellow", 1), c("purple", 2), c("brown", 3)]  # identical strength
    assert claim_is_provable(mine, theirs, [], tiebreak_mine=True)
    assert not claim_is_provable(mine, theirs, [], tiebreak_mine=False)


def test_unbeatable_formation_is_provable_against_empty_side():
    mine = [c("red", 7), c("red", 8), c("red", 9)]
    pool = unseen_pool(table=mine, hand=[])
    assert claim_is_provable(mine, [], pool, tiebreak_mine=True)


def test_beatable_formation_is_not_provable():
    mine = [c("red", 1), c("red", 2), c("red", 3)]
    pool = unseen_pool(table=mine, hand=[])
    assert not claim_is_provable(mine, [], pool, tiebreak_mine=True)


def test_ties_do_not_block_a_claim_when_claimer_completed_first():
    mine = [c("blue", 7), c("blue", 8), c("blue", 9)]
    theirs = [c("red", 8), c("red", 9)]
    table = mine + theirs
    pool = unseen_pool(table, hand=[])  # red 7 would tie, never beat
    assert claim_is_provable(mine, theirs, pool, tiebreak_mine=True)
    assert not claim_is_provable(mine, theirs, pool, tiebreak_mine=False)


def test_own_hand_is_removed_from_the_pool():
    mine = [c("blue", 6), c("blue", 7), c("blue", 8)]  # color run, total 21
    theirs = [c("red", 7), c("red", 8)]
    table = mine + theirs
    # red 9 would give them 7-8-9 and beat me...
    assert not claim_is_provable(mine, theirs, unseen_pool(table, hand=[]), tiebreak_mine=True)
    # ...but if I hold it, they cannot (red 6 only ties)
    assert claim_is_provable(
        mine, theirs, unseen_pool(table, hand=[c("red", 9)]), tiebreak_mine=True
    )


def test_opponent_that_can_never_complete_loses_the_stone():
    mine = [c("red", 1), c("blue", 3), c("green", 5)]
    theirs = [c("red", 9)]
    assert claim_is_provable(mine, theirs, pool=[c("brown", 9)], tiebreak_mine=True)  # needs 2 cards


def test_win_conditions():
    assert check_win([0, 0, 0, None, None, None, None, None, None]) == (0, "three_adjacent")
    assert check_win([0, 1, 0, 1, 0, 1, 0, 0, None]) == (0, "five_stones")  # 0,2,4,6,7: no three in a row
    assert check_win([0, 0, 1, 0, 0, 1, 1, None, None]) is None
    # simultaneous wins go to the first seat in `order`
    both = [0, 0, 0, 1, 1, 1, None, None, None]
    assert check_win(both, order=(1, 0)) == (1, "three_adjacent")
