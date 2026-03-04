import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import engine

# constants (must match engine.py)
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

START_STACK = 400
N_ACTIONS = 4  # FOLD, CHECK, CALL, RAISE
N_ROUNDS = 4   # preflop, flop, turn, river
MAX_ACTIONS_PER_ROUND = 6  # paper: up to 6 sequential actions per round

# Bet features: per position (bet_occurred, bet_size) + continue_cost + pot
# = (MAX_ACTIONS_PER_ROUND * N_ROUNDS * 2) + 2 = 50
N_BET_FEATS = MAX_ACTIONS_PER_ROUND * N_ROUNDS * 2 + 2

# action index mapping for network output engine action
ACTION_LIST = [FOLD, CHECK, CALL, RAISE]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}
ACTION_STRINGS = ['Fold', 'Check', 'Call', 'Raise']


# state encoding

def encode_state(state, player):
    """
    Encode a single game state from `player`'s perspective.

    Args:
        state: (button, street, pip0, pip1, cards, action_seq, bet_sizes, raises, action_count)
        player: 0 or 1 — which player's infoset to encode

    Returns:
        card_groups: tuple of 4 LongTensors (hole[2], flop[3], turn[1], river[1])
        bet_feats:   FloatTensor of shape (N_BET_FEATS,)
    """
    street = state[1]
    pip0 = state[2]
    pip1 = state[3]
    cards = state[4]
    action_seq = state[5]    # shape (4, MAX_ACTIONS_PER_ROUND)
    bet_sizes_hist = state[6]  # shape (4, MAX_ACTIONS_PER_ROUND)

    # cards
    if player == 0:
        hole = torch.tensor(cards[:2], dtype=torch.long)
    else:
        hole = torch.tensor(cards[2:4], dtype=torch.long)

    if street >= FLOP:
        flop = torch.tensor(cards[4:7], dtype=torch.long)
    else:
        flop = torch.full((3,), -1, dtype=torch.long)

    if street >= TURN:
        turn = torch.tensor(cards[7:8], dtype=torch.long)
    else:
        turn = torch.full((1,), -1, dtype=torch.long)

    if street >= RIVER:
        river = torch.tensor(cards[8:9], dtype=torch.long)
    else:
        river = torch.full((1,), -1, dtype=torch.long)

    # bet features (paper Section 5.1)
    # Each of N_ROUNDS rounds has up to MAX_ACTIONS_PER_ROUND positions.
    # Each position is encoded as (bet_occurred: binary, bet_size: float).
    # This gives 6 * 4 * 2 = 48 positional features + 2 extras = 50 total.

    bet_occurred = (action_seq != 0).astype(np.float32).flatten()  # (24,)
    bet_size = bet_sizes_hist.flatten().astype(np.float32)          # (24,)

    # Continue cost and pot, normalized by stack
    if player == 0:
        my_pip = float(pip0)
        opp_pip = float(pip1)
    else:
        my_pip = float(pip1)
        opp_pip = float(pip0)

    continue_cost = abs(opp_pip - my_pip) / START_STACK
    pot = (my_pip + opp_pip) / (2.0 * START_STACK)

    bet_feats = torch.tensor(
        np.concatenate([bet_size, bet_occurred, [continue_cost, pot]]),
        dtype=torch.float32,
    )

    return (hole, flop, turn, river), bet_feats


def batch_encode_states(states, players):
    """
    Batch-encode a list of (state, player) pairs.

    Returns:
        card_groups: tuple of 4 LongTensors, each (B, n_cards_in_group)
        bet_feats:   FloatTensor (B, N_BET_FEATS)
    """
    all_cards = [[], [], [], []]  # hole, flop, turn, river
    all_bets = []

    for state, player in zip(states, players):
        card_groups, bet_feats = encode_state(state, player)
        for i, cg in enumerate(card_groups):
            all_cards[i].append(cg)
        all_bets.append(bet_feats)

    card_groups = tuple(torch.stack(cg) for cg in all_cards)
    bet_feats = torch.stack(all_bets)
    return card_groups, bet_feats


# network modules

class CardEmbedding(nn.Module):
    """
    From the paper (Appendix C):
    Each card is represented as the sum of three learned embeddings:
    rank (0-12), suit (0-3), and card identity (0-51).
    Cards within a group (hole, flop, etc.) are summed.
    Invalid cards (index -1) are zeroed out.
    """

    def __init__(self, dim):
        super().__init__()
        self.rank = nn.Embedding(13, dim)
        self.suit = nn.Embedding(4, dim)
        self.card = nn.Embedding(52, dim)

    def forward(self, x):
        """
        x: LongTensor (B, num_cards) with values in [0..51] or -1
        Returns: (B, dim)
        """
        B, num_cards = x.shape
        x_flat = x.view(-1)                          # (B * num_cards,)
        valid = x_flat.ge(0).float()                  # 1 where card exists
        x_clamped = x_flat.clamp(min=0)               # safe index

        embs = (
            self.card(x_clamped)
            + self.rank(x_clamped // 4)
            + self.suit(x_clamped % 4)
        )
        embs = embs * valid.unsqueeze(1)              # zero out missing cards
        return embs.view(B, num_cards, -1).sum(dim=1) # sum across cards (B, dim)


class DeepCFRNetwork(nn.Module):
    """
    Architecture from Brown et al. 2018:

    Card branch:  4 CardEmbeddings, concat, 3 FC layers
    Bet branch:   bet features, 2 FC layers (skip on layer 2)
    Combined:     concat, 3 FC layers (skip on layers 2, 3), normalize, action head

    Used for both the value (advantage) network and the average strategy network.
    """

    def __init__(self, n_actions=N_ACTIONS, n_bet_feats=N_BET_FEATS, dim=64, zeros=False):
        """
        Args:
            n_actions:  number of output heads
            n_bet_feats: bet feature vector size (paper: 6*4*2 + 2 = 50)
            dim:        hidden dimension (paper uses 256 for HULH; 64 is fine for training)
        """
        super().__init__()

        # card branch
        self.card_embeddings = nn.ModuleList([
            CardEmbedding(dim),  # hole  (2 cards)
            CardEmbedding(dim),  # flop  (3 cards)
            CardEmbedding(dim),  # turn  (1 card)
            CardEmbedding(dim),  # river (1 card)
        ])
        self.card1 = nn.Linear(dim * 4, dim)
        self.card2 = nn.Linear(dim, dim)
        self.card3 = nn.Linear(dim, dim)

        # bet branch
        self.bet1 = nn.Linear(n_bet_feats, dim)
        self.bet2 = nn.Linear(dim, dim)

        # combined trunk
        self.comb1 = nn.Linear(2 * dim, dim)
        self.comb2 = nn.Linear(dim, dim)
        self.comb3 = nn.Linear(dim, dim)

        # output
        self.action_head = nn.Linear(dim, n_actions)

        # initialize all outputs to zero for first epoch (from paper section 5.1)
        if zeros:
            nn.init.zeros_(self.action_head.weight)
            nn.init.zeros_(self.action_head.bias)

    def forward(self, card_groups, bet_feats):
        """
        Args:
            card_groups: tuple of 4 LongTensors (B,2), (B,3), (B,1), (B,1)
            bet_feats:   FloatTensor (B, n_bet_feats)

        Returns:
            (B, n_actions) — advantages (value net) or logits (strategy net)
        """
        # card branch, no skips
        card_embs = [emb(cg) for emb, cg in zip(self.card_embeddings, card_groups)]
        x = torch.cat(card_embs, dim=1)  # (B, dim*4)
        x = F.relu(self.card1(x))
        x = F.relu(self.card2(x))
        x = F.relu(self.card3(x))

        # bet branch (skip on layer 2)
        y = F.relu(self.bet1(bet_feats))
        y = F.relu(self.bet2(y) + y)

        # combined trunk (skip on layers 2, 3)
        z = torch.cat([x, y], dim=1)     # (B, 2*dim)
        z = F.relu(self.comb1(z))
        z = F.relu(self.comb2(z) + z)
        z = F.relu(self.comb3(z) + z)

        # normalize (zero mean, unit variance — from paper section 5.1)
        z = (z - z.mean(dim=1, keepdim=True)) / (z.std(dim=1, keepdim=True) + 1e-6)

        return self.action_head(z)


# ── regret matching ───────────────────────────────────────────

def regret_match(advantages):
    """
    Convert predicted advantages to a strategy via regret matching.

    From paper section 2.1:
    - Standard RM plays proportional to positive regrets
    - Paper modification: when all regrets <= 0, play the highest regret
      action with probability 1 (not uniform like normal RM

    Args:
        advantages: tensor (n_actions,) or (B, n_actions)

    Returns:
        strategy: same shape, probability distribution over actions
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    pos = F.relu(advantages)
    total = pos.sum(dim=1, keepdim=True)

    # When sum of positive regrets > 0, play proportionally
    strategy = pos / total.clamp(min=1e-8)

    all_neg = (total.squeeze(1) == 0)
    if all_neg.any():
        adv_sub = advantages[all_neg]
        max_vals = adv_sub.max(dim=1).values
        all_zero = (max_vals == 0)

        # All zero, uniform over legal actions
        if all_zero.any():
            legal = adv_sub[all_zero] > -float('inf')
            uniform = legal.float()
            uniform = uniform / uniform.sum(dim=1, keepdim=True)
            strategy[all_neg.nonzero(as_tuple=True)[0][all_zero]] = uniform

        # All negative, use argmax
        all_negative = ~all_zero
        if all_negative.any():
            best = adv_sub[all_negative].argmax(dim=1)
            fallback = torch.zeros_like(adv_sub[all_negative])
            fallback.scatter_(1, best.unsqueeze(1), 1.0)
            strategy[all_neg.nonzero(as_tuple=True)[0][all_negative]] = fallback

    if squeeze:
        strategy = strategy.squeeze(0)
    return strategy


# ── quick test ────────────────────────────────────────────────

if __name__ == "__main__":
    model = DeepCFRNetwork(n_actions=N_ACTIONS, dim=256)
    statedict = torch.load('Strat_50.pt', map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(statedict)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Simulate a single state
    cards = np.arange(52)
    np.random.shuffle(cards)
    Hole_cards = ['2d', '7s']  # player 0 hole cards
    cards[0] = engine.string_to_int_card(Hole_cards[0])  #
    cards[1] = engine.string_to_int_card(Hole_cards[1])  #
  
    #print(cards[0:2])

    action_seq = np.zeros((4, MAX_ACTIONS_PER_ROUND), dtype=np.float32)
    bet_sizes_hist = np.zeros((4, MAX_ACTIONS_PER_ROUND), dtype=np.float32)
    state = (0, PREFLOP, 2, 4, cards, action_seq, bet_sizes_hist, 0, 0)

    card_groups, bet_feats = encode_state(state, player=0)

    # Add batch dim
    card_groups_b = tuple(cg.unsqueeze(0) for cg in card_groups)
    bet_feats_b = bet_feats.unsqueeze(0)

    with torch.no_grad():
        legal_actions = engine.get_legal_actions(state)
        advantages = model(card_groups_b, bet_feats_b)
        mask = torch.tensor([a in legal_actions for a in ACTION_LIST], dtype=torch.bool)
        masked_advantages = torch.where(
                mask, advantages, torch.tensor(-float("inf"))
            )
        strategy = regret_match(masked_advantages)

    """print(f"Advantages: {advantages.squeeze(0).numpy()}")
    print(f"Strategy:   {strategy.numpy()}")
    print(f"Actions:    {ACTION_LIST}")
    print(f"Bet feats size: {bet_feats.shape[0]}")"""

    print("----------PREFLOP STRATEGY----------")
    print(f"Hole cards: {Hole_cards}")
    print("Strategy:")
    for i, action in enumerate(strategy.numpy()[0]):
        if action != 0:
            print(f"{ACTION_STRINGS[i]}: {int(action * 1000)/10}%")
    
