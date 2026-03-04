# Deep-CFR-for-HULHE
A Deep CFR poker bot for heads-up 4/8 limit Texas Hold'em.

Based on Noam Brown's Deep CFR architecture (https://arxiv.org/pdf/1811.00164).
Game engine with inbuilt betting abstraction and efficient state encoding from scratch

engine.py contains all of the game logic and encodes states as a tuple. The main logic controlled by the "proceed" method takes a state vector and an action, and returns the new state vector or payoffs if the hand is done. phevaluator used for high speed hand evaluation using a perfect hashing algorithm. Configurable game parameters (blinds, stacks, etc) contained within global variables at the start of the file.

network.py contains the network architecture used for the Strategy and Advantage networks. As written, they are exactly the same as in the paper, with a dim of 256 used for training.

traversal.py contains the cfr traversal logic. Almost identical to MCCFR traversal, but uses slightly modified regret matching, and uses advantage network to replace cumulative regret table.
