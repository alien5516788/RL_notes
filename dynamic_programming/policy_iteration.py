from math import inf

from ..reinforcement_learning.mdp import (
    GAMMA,
    Action,
    State,
    actions,
    reward,
    states,
    world_model,
)

# Policy (π) - learnt
#
#   π(a | s)
#
pi_table: dict[tuple[State, Action], float] = {}


def policy(s: State, a: Action) -> float:
    return pi_table[(s, a)]


# Value (V) - learnt
#
#  V(s) -> v
#
v_table: dict[State, float] = {}


def value(s: State) -> float:
    return v_table[s]


# Policy Evaluation
# Value is improved by following a current policy
def policy_evaluation():
    for s in states:
        # v
        v = 0.0

        # Σ_a
        for a in actions:
            # π(a | s)
            p = policy(s, a)

            # Σ_s'
            for s_next in states:
                # v += π(a | s) * Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
                v += (
                    p
                    * world_model(s, a, s_next)
                    * (reward(s, a, s_next) + (GAMMA * value(s_next)))
                )

        # V(s) <- v
        v_table[s] = v


# Policy Improvement
#
#     Q(s, a) = Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
#     a* = argmax_a Q(s, a)
#     π(a | s) <- 1 if a == a*; 0 otherwise                    // greedy/deterministic
#     π(a | s) <- 0.9 if a == a*; 0.1 * (1 / n) otherwise      // stochastic
#
# Policy is improved by calculating the Q(s,a) for every possible action a and
#   updating the policy table with the action a* (action a with best Q value)
#
def policy_improvement():
    for s in states:
        # Place holders until calculated
        best_action = actions[0]  # a*
        best_action_value = -inf  # max_a Q(s, a)

        for a in actions:
            # q
            q = 0.0

            # Σ_s'
            for s_next in states:
                # q += P(s' | s, a) * [R(s, a, s') + γV(s')]
                q += world_model(s, a, s_next) * (
                    reward(s, a, s_next) + (GAMMA * value(s_next))
                )

            # Q(s, a) = q

            #  a* <- arg_max_a Q(s, a)
            if q > best_action_value:
                best_action = a  # a*
                best_action_value = q  # max_a Q(s, a)

        # π(a | s) <- 1 if a == a*, 0 otherwise
        for a in actions:
            if a == best_action:
                pi_table[(s, a)] = 1.0
            else:
                pi_table[(s, a)] = 0.0


def policy_iteration():
    # Policy Evaluation
    converged = False

    while not converged:
        policy_evaluation()

    # Policy Improvement
    policy_improvement()


def train():
    converged = False

    while not converged:
        policy_iteration()
