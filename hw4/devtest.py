import gymnasium
import numpy as np

def test():
    """
    Tests that the QLearning implementation successfully learns the
    FrozenLake-v1 environment.
    """
    from src import QLearning
    from src.random import rng
    rng.seed()

    # https://gymnasium.farama.org/environments/toy_text/frozen_lake/
    env = gymnasium.make('FrozenLake-v1')
    env.reset()

    agent = QLearning(epsilon=0.4, gamma=0.9, alpha=0.5)
    state_action_values, rewards = agent.fit(env, steps=10000)

    state_values = np.max(state_action_values, axis=1)

    assert state_action_values.shape == (16, 4)
    assert len(rewards) == 100

    print('state values:')
    print(state_values[np.array([5, 7, 11, 12, 15])])
    print(state_values[np.array([0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14])])

    assert np.allclose(state_values[np.array([5, 7, 11, 12, 15])], np.zeros(5))
    assert np.all(state_values[np.array([0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14])] > 0)

if __name__ == '__main__':
    test()
