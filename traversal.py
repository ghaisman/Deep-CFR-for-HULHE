import torch
import numpy as np

import engine
import network

# constants
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

START_STACK = 400
N_ACTIONS = 4  # FOLD, CHECK, CALL, RAISE

# action index mapping for network output -> engine action
ACTION_LIST = [FOLD, CHECK, CALL, RAISE]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}


def Traverse(state, traverser, theta0, theta1, M0, M1, strat_memory, iteration):
    # Terminal: fold returned reward from player 0's perspective
    if not isinstance(state, tuple) or len(state) != 9:
        return state if traverser == 0 else -state

    button, street, pip0, pip1, cards, action_seq, bet_sizes, raises, action_count = state

    # Terminal: showdown evaluate and return reward from traverser's perspective
    if street == SHOWDOWN:
        reward = engine.proceed(state, None)  # triggers showdown evaluation
        return reward if traverser == 0 else -reward

    legal_actions = engine.get_legal_actions(state)
    if len(legal_actions) == 0:
        raise ValueError("No legal actions available, invalid state reached.")

    # Legal action mask over full action space
    mask = torch.tensor([a in legal_actions for a in ACTION_LIST], dtype=torch.bool)

    if button == traverser:
        # Traverser's decision node — use traverser's value network
        model = theta0 if traverser == 0 else theta1
        memory = M0 if traverser == 0 else M1

        card_groups, bet_feats = network.encode_state(state, player=traverser)
        card_groups_b = tuple(cg.unsqueeze(0) for cg in card_groups)
        bet_feats_b = bet_feats.unsqueeze(0)

        with torch.no_grad():
            advantages = model(card_groups_b, bet_feats_b).squeeze(0)
            masked_advantages = torch.where(
                mask, advantages, torch.tensor(-float("inf"))
            )
            strategy = network.regret_match(masked_advantages)
        strategy_np = strategy.numpy()

        # Traverse all actions
        value = np.zeros(N_ACTIONS)
        for action in legal_actions:
            nextstate = engine.proceed(state, action)
            value[action] = Traverse(
                nextstate, traverser, theta0, theta1, M0, M1, strat_memory, iteration
            )

        node_value = np.sum(strategy_np * value)

        # Compute instantaneous regrets (0 for illegal actions)
        regret = np.zeros(N_ACTIONS)
        for action in legal_actions:
            regret[action] = value[action] - node_value

        memory.append(
            (
                card_groups,
                bet_feats,
                mask,
                torch.tensor(regret, dtype=torch.float32),
                iteration,
            )
        )
        return node_value
    else:
        # Opponent's decision node — use opponent's value network
        model = theta1 if traverser == 0 else theta0

        card_groups, bet_feats = network.encode_state(state, player=button)
        card_groups_b = tuple(cg.unsqueeze(0) for cg in card_groups)
        bet_feats_b = bet_feats.unsqueeze(0)

        with torch.no_grad():
            advantages = model(card_groups_b, bet_feats_b).squeeze(0)
            masked_advantages = torch.where(
                mask, advantages, torch.tensor(-float("inf"))
            )
            strategy = network.regret_match(masked_advantages)
        strategy_np = strategy.numpy()

        # Sample one action
        action = np.random.choice(ACTION_LIST, p=strategy_np)

        # Store strategy sample with legal mask and iteration weight for linear CFR
        strat_memory.append(
            (
                card_groups,
                bet_feats,
                mask,
                torch.tensor(strategy_np, dtype=torch.float32),
                iteration,
            )
        )

        next_state = engine.proceed(state, action)
        return Traverse(
            next_state, traverser, theta0, theta1, M0, M1, strat_memory, iteration
        )
