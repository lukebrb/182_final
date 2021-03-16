import numpy as np
from environment import State, Environment, Action
from collections import defaultdict
from typing import Any
import sys


def SARSA(
    Q: defaultdict[Any, list[int]], epsilon: float, alpha: float, gamma: float, n: int
) -> tuple[list[State], defaultdict[Any, list[int]]]:
    """ Runs an episode of n-step SARSA

    Args:
        Q ([type]): [description]
        pi ([type]): A multi-dimensional array of policies
        epsilon ([type]): [description]
        alpha ([type]): [description]
        n_steps ([type]): [description]
    """
    States = []
    Actions = []
    Rewards = []

    env = Environment.with_csv_fetcher()
    # Initialize and store S_0 != terminal
    S_0 = env.default_state()
    States.append(S_0)

    pi = (
        lambda S: np.argmax(Q[S])
        if np.random.random() > epsilon
        else np.random.choice(len(Q[S]))
    )

    # Select and store an action $A_0 ~ \pi ( \dot | S_0)$
    A_0 = pi(S_0)
    Actions.append(A_0)

    T = 130000
    t = 0

    tau = 0

    S_t = S_0
    A_t = A_0

    while tau < T - 1:
        if t < T:
            # Take action A_t
            S_t1, R_t1 = env.step(States[t], Actions[t])

            States.append(S_t1)
            Rewards.append(R_t1)

            if S_t1.is_terminal:
                T = t + 1
            else:
                # Select and store an action A_t+1 ~ pi(. | S_t+1)
                A_t1 = pi(S_t1)
                Actions.append(A_t1)

        tau = t - n + 1
        if tau >= 0:
            G = np.sum(
                [
                    (gamma ** (i - tau - 1)) * Rewards[i]
                    for i in range(tau + 1, min(tau + n, T))
                ]
            )

            if tau + n < T:
                G = G + ((gamma ** n) * Q[States[tau + n]][int(Actions[tau + n])])

            Q[States[tau]][int(Actions[tau])] = Q[States[tau]][int(Actions[tau])] + (alpha * (G - Q[States[tau]][int(Actions[tau])]))

            # If \pi is being learned, then ensure that \pi ( * | S_tau) is \epsilon-greedy wrt Q

        t += 1

    return (States, Q)


def DynaQ(Q, model, alpha, gamma, epsilon, n_planning_steps, max_steps):
    env = Environment.with_csv_fetcher()
    S = env.default_state()

    steps = max_steps

    for step in range(max_steps):
        A = (
            np.argmax(Q[S])
            if np.random.random() > epsilon
            else np.random.choice(len(Q[S]))
        )
        S_next, R = env.step(S, A)
        model[S][A] = [S_next, R]

        Q[S][A] = Q[S][A] + alpha * (R + (gamma * (np.max(Q[S_next]))) - Q[S][A])

        for k in range(n_planning_steps):
            state_p = np.random.choice([state_key for state_key in model.keys() if len(model[state_key].values()) > 0]) 
            action_p = np.random.choice(list(model[state_p].keys()))

            state_next_p, r_p = model[state_p][action_p]

            Q[state_p][action_p] = (
                Q[state_p][action_p]
                + alpha * (r_p + (gamma * np.max(Q[state_next_p])))
                - Q[state_p][action_p]
            )

        if S_next == None:
            steps = step
            break
        state = S_next
    return Q, model, steps

