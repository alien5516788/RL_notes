from ..reinforcement_learning.mdp import (
    GAMMA,
    Action,
    State,
    actions,
    reward,
    states,
    world_model,
)


# Policy (π) - defined
#
#   π(a | s)
#
def policy(s: State, a: Action) -> float:
    return 0.9


# Value (V) - learnt
#
#  V(s) -> v
#
v_table: dict[State, float] = {}


def value(s: State) -> float:
    return v_table[s]


# Value Backup / Bellman equation for Policy Evaluation
#
#   V(s) <- Σ_a π(a | s) * Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
#
# Value is learnt by continuously interacting with the world_model
#
def policy_evaluation():
    for s in states:
        # v
        v = 0.0
        # Σ_a
        for a in actions:
            # π(a | s)
            action_probability = policy(s, a)
            # Σ_s'
            for s_next in states:
                # P(s' | s, a)
                transition_probability = world_model(s, a, s_next)
                # R(s, a, s')
                immediate_reward = reward(s, a, s_next)
                # V(s')
                future_value = value(s_next)
                # v += π(a | s) * Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
                v += (
                    action_probability
                    * transition_probability
                    * (immediate_reward + (GAMMA * future_value))
                )

        # V(s) <- v
        v_table[s] = v


def train():
    converged = True

    while not converged:
        policy_evaluation()
