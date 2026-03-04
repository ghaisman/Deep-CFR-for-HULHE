import numpy as np
import time
from phevaluator.evaluator import evaluate_cards
import random

########## 4/8 LIMIT HOLDEM ENGINE

SB = 2
BB = 4

PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
SHOWDOWN = 4

# Unified action constants
FOLD = 0
CHECK = 1
CALL = 2
RAISE = 3

SMALL_BET = 4
BIG_BET = 8

SUITS = {'c': 0, 'd': 1, 'h': 2, 's': 3}
RANKS = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}

# Per-street raise limits (Appendix A: max 3 raises rounds 1-2, max 4 rounds 3-4)
RAISES_LIMIT_BY_STREET = {
    PREFLOP: 3,
    FLOP: 3,
    TURN: 4,
    RIVER: 4,
}

# Max actions per betting round (paper: up to 6 sequential actions per round)
MAX_ACTIONS_PER_ROUND = 6

START_STACK = 400


def init_state_vector():
    """
    Returns initialized state vector with dealt cards etc.
    History now tracks per-position action sequences for full bet encoding.
    history shape: (4, MAX_ACTIONS_PER_ROUND) — one row per street,
    each entry is the action taken at that position (0 = no action yet).
    We use a separate encoding: 0 = empty, 1 = fold, 2 = check, 3 = call, 4 = raise.
    Plus bet sizes: (4, MAX_ACTIONS_PER_ROUND) floats for the bet size at each position.
    """
    cards = np.arange(52)
    np.random.shuffle(cards)
    button = 0
    street = 0
    raises = 0
    # action_seq: which action was taken at each position (0=empty)
    action_seq = np.zeros((4, MAX_ACTIONS_PER_ROUND), dtype=np.float32)
    # bet_sizes: the bet size (as fraction of pot or raw) at each position
    bet_sizes = np.zeros((4, MAX_ACTIONS_PER_ROUND), dtype=np.float32)
    # action_count: how many actions have been taken on the current street
    action_count = 0
    state_vector = (button, street, SB, BB, cards, action_seq, bet_sizes, raises, action_count)
    return state_vector


actions = [FOLD, CALL, CHECK, RAISE]


def proceed(state, action):
    """
    Takes current state and action, returns new state or reward if hand ended.
    """
    oldbutton = state[0]
    oldstreet = state[1]
    oldpip0 = state[2]
    oldpip1 = state[3]
    cards = state[4]
    old_action_seq = state[5]
    old_bet_sizes = state[6]
    oldraises = state[7]
    old_action_count = state[8]

    # Record action in history
    action_seq = old_action_seq.copy()
    bet_sizes_hist = old_bet_sizes.copy()

    # Encode action into history (1=fold, 2=check, 3=call, 4=raise)
    action_code = action + 1  # shift so 0 remains "empty"
    pos = min(old_action_count, MAX_ACTIONS_PER_ROUND - 1)
    action_seq[oldstreet, pos] = action_code

    # Record bet size as fraction of pot for raises
    pot = oldpip0 + oldpip1
    if action == RAISE:
        if oldstreet <= FLOP:
            bet_sizes_hist[oldstreet, pos] = SMALL_BET / max(pot, 1)
        else:
            bet_sizes_hist[oldstreet, pos] = BIG_BET / max(pot, 1)
    elif action == CALL:
        call_amount = abs(oldpip0 - oldpip1)
        bet_sizes_hist[oldstreet, pos] = call_amount / max(pot, 1)

    new_action_count = old_action_count + 1

    if oldstreet == PREFLOP:  # preflop
        if action == FOLD:
            return -oldpip0 if oldbutton == 0 else oldpip1
        elif action == CALL:
            if oldbutton == 0:
                if oldpip1 == BB:
                    next_state = (1, PREFLOP, BB, BB, cards, action_seq, bet_sizes_hist, oldraises, new_action_count)
                else:
                    next_state = (1, FLOP, oldpip1, oldpip1, cards, action_seq, bet_sizes_hist, 0, 0)
            else:
                next_state = (1, FLOP, oldpip0, oldpip0, cards, action_seq, bet_sizes_hist, 0, 0)
        elif action == CHECK:
            next_state = (1, FLOP, BB, BB, cards, action_seq, bet_sizes_hist, 0, 0)
        elif action == RAISE:
            if oldbutton == 0:
                pip_0 = oldpip1 + SMALL_BET
                pip_1 = oldpip1
            else:
                pip_1 = oldpip0 + SMALL_BET
                pip_0 = oldpip0
            next_state = ((1 - oldbutton), oldstreet, pip_0, pip_1, cards, action_seq, bet_sizes_hist, oldraises + 1, new_action_count)

    elif oldstreet == FLOP:  # postflop
        if action == FOLD:
            return -oldpip0 if oldbutton == 0 else oldpip1
        elif action == CALL:
            if oldbutton == 0:
                next_state = (1, TURN, oldpip1, oldpip1, cards, action_seq, bet_sizes_hist, 0, 0)
            else:
                next_state = (1, TURN, oldpip0, oldpip0, cards, action_seq, bet_sizes_hist, 0, 0)
        elif action == CHECK:
            if oldbutton == 0:  # proceed to turn
                next_state = (1, TURN, oldpip0, oldpip1, cards, action_seq, bet_sizes_hist, 0, 0)
            else:  # stay on same street
                next_state = (0, oldstreet, oldpip0, oldpip1, cards, action_seq, bet_sizes_hist, 0, new_action_count)
        elif action == RAISE:
            if oldbutton == 0:
                pip_0 = oldpip1 + SMALL_BET
                pip_1 = oldpip1
            else:
                pip_1 = oldpip0 + SMALL_BET
                pip_0 = oldpip0
            next_state = ((1 - oldbutton), oldstreet, pip_0, pip_1, cards, action_seq, bet_sizes_hist, oldraises + 1, new_action_count)

    elif oldstreet == TURN:  # postturn
        if action == FOLD:
            return -oldpip0 if oldbutton == 0 else oldpip1
        elif action == CALL:
            if oldbutton == 0:
                next_state = (1, RIVER, oldpip1, oldpip1, cards, action_seq, bet_sizes_hist, 0, 0)
            else:
                next_state = (1, RIVER, oldpip0, oldpip0, cards, action_seq, bet_sizes_hist, 0, 0)
        elif action == CHECK:
            if oldbutton == 0:  # proceed to river
                next_state = (1, RIVER, oldpip0, oldpip1, cards, action_seq, bet_sizes_hist, 0, 0)
            else:  # stay on same street
                next_state = (0, oldstreet, oldpip0, oldpip1, cards, action_seq, bet_sizes_hist, 0, new_action_count)
        elif action == RAISE:
            if oldbutton == 0:
                pip_0 = oldpip1 + BIG_BET
                pip_1 = oldpip1
            else:
                pip_1 = oldpip0 + BIG_BET
                pip_0 = oldpip0
            next_state = ((1 - oldbutton), oldstreet, pip_0, pip_1, cards, action_seq, bet_sizes_hist, oldraises + 1, new_action_count)

    elif oldstreet == RIVER:  # postriver
        if action == FOLD:
            return -oldpip0 if oldbutton == 0 else oldpip1
        elif action == CALL:
            if oldbutton == 0:
                next_state = (1, SHOWDOWN, oldpip1, oldpip1, cards, action_seq, bet_sizes_hist, 0, 0)
            else:
                next_state = (1, SHOWDOWN, oldpip0, oldpip0, cards, action_seq, bet_sizes_hist, 0, 0)
        elif action == CHECK:
            if oldbutton == 0:  # second to act checks -> showdown
                next_state = (1, SHOWDOWN, oldpip0, oldpip1, cards, action_seq, bet_sizes_hist, 0, 0)
            else:  # first to act checks -> opponent acts
                next_state = (0, oldstreet, oldpip0, oldpip1, cards, action_seq, bet_sizes_hist, 0, new_action_count)
        elif action == RAISE:
            if oldbutton == 0:
                pip_0 = oldpip1 + BIG_BET
                pip_1 = oldpip1
            else:
                pip_1 = oldpip0 + BIG_BET
                pip_0 = oldpip0
            next_state = ((1 - oldbutton), oldstreet, pip_0, pip_1, cards, action_seq, bet_sizes_hist, oldraises + 1, new_action_count)

    if next_state[1] == SHOWDOWN:  # showdown
        # evaluate hands and return reward
        assert next_state[2] == next_state[3], f"{next_state[2]} vs {next_state[3]} at showdown, should be equal"
        hand0 = cards[:2].astype(object)
        hand1 = cards[2:4].astype(object)
        board = cards[4:9].astype(object)
        # evaluate hand strength and determine winner
        hand0_strength = evaluate_cards(*hand0, *board)
        hand1_strength = evaluate_cards(*hand1, *board)
        if hand0_strength < hand1_strength:
            return oldpip1
        elif hand1_strength < hand0_strength:
            return -oldpip0
        else:
            return 0
    if not isinstance(next_state, tuple) or len(next_state) != 9:
        raise ValueError(f"Invalid next state returned: {next_state}")
    return next_state


def get_legal_actions(state):
    pip0 = state[2]
    pip1 = state[3]
    street = state[1]
    raises = state[7]
    to_call = max(pip0, pip1) - min(pip0, pip1)

    if to_call == 0:
        legal = [CHECK]
    else:
        legal = [FOLD, CALL]

    # Per-street raise limit
    raises_limit = RAISES_LIMIT_BY_STREET.get(street, 4)
    if raises < raises_limit:
        legal.append(RAISE)

    return legal


def string_to_int_card(card):
    rank = RANKS[card[0]]
    suit = SUITS[card[1]]
    return rank * 4 + suit


"""start_time = time.time()

for i in range(1_000):
    start = init_state_vector()
    proceeded = proceed(start, CALL)

end_time = time.time()

print("Steps/s: ", 1_000 / (end_time - start_time))

#######SIMULATE 1000 hands with random actions#######

start_time = time.time()
for i in range(1_000):
    s = init_state_vector()
    done = False
    while not done:
        legal_actions = get_legal_actions(s)
        action = random.choice(legal_actions)
        result = proceed(s, action)
        if isinstance(result, tuple):
            s = result
        else:
            done = True
end_time = time.time()
print("Hands/s: ", 1_000 / (end_time - start_time))"""
