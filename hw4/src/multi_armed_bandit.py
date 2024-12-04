import numpy as np
import src.random


class MultiArmedBandit:
    """
    MultiArmedBandit reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
    """

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def fit(self, env, steps=1000, num_bins=100):
        """
        Trains the MultiArmedBandit on an OpenAI Gymnasium environment.

        See page 32 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2020.pdf).
        Initialize your parameters as all zeros. For the step size, use
        1/N, where N is the number of times the current action has been
        performed. (This is the version of Bandits we saw in lecture before
        we introduced alpha). Use an epsilon-greedy approach to pick actions.

        See (https://gymnasium.farama.org/) for examples of how to use the OpenAI
        Gymnasium Environment interface.

        In every step of the fit() function, you should sample
            two random numbers using functions from `src.random`.
            1.  First, use either `src.random.rand()` or `src.random.uniform()`
                to decide whether to explore or exploit.
            2. Then, use `src.random.choice` or `src.random.randint` to decide
                which action to choose. Even when exploiting, you should make a
                call to `src.random` to break (possible) ties.

        Please don't use `np.random` functions; use the ones from `src.random`!
        Please do not use `env.action_space.sample()`!

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "terminated or truncated" returned
            from env.step() is True.
          - When choosing to exploit the best action rather than exploring,
            do not use np.argmax: it will deterministically break ties by
            choosing the lowest index of among the tied values. Instead,
            please *randomly choose* one of those tied-for-the-largest values.
          - MultiArmedBandit treats all environment states the same. However,
            in order to have the same API as agents that model state, you must
            explicitly return the state-action-values Q(s, a). To do so, just
            copy the action values learned by MultiArmedBandit S times, where
            S is the number of states.

        Arguments:
          env - (Env) An OpenAI Gymnasium environment with discrete actions and
            observations. See the OpenAI Gymnasium documentation for example use
            cases (https://gymnasium.farama.org/api/env/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length `num_bins`.
            Let s = int(np.ceil(steps / `num_bins`)), then rewards[0] should
            contain the average reward over the first s steps, rewards[1]
            should contain the average reward over the next s steps, etc.
            Please note that: The total number of steps will not always divide evenly by the 
            number of bins. This means the last group of steps may be smaller than the rest of 
            the groups. In this case, we can't divide by s to find the average reward per step 
            because we have less than s steps remaining for the last group.
        """
        # env = env.unwrapped

        # set up Q function, rewards
        n_actions, n_states = env.action_space.n, env.observation_space.n
        self.Q = np.zeros(n_actions)
        self.N = np.zeros(n_actions)
        # 
        avg_rewards = np.zeros([num_bins])
        s = int(np.ceil(steps/num_bins))
        all_rewards = []
        # reset environment before your first action
        env.reset()
        for step in range(steps): # TODO idk if this is right
            # decide to explore or exploit
            sample = src.random.rand()
            # get action
            if sample < self.epsilon: # explore
              chosen_action = src.random.choice([i for i in range(n_actions)])
            elif sample >= self.epsilon: # exploit
              best_val = -float('inf')
              for action, val in enumerate(self.Q):
                if val > best_val:
                  best_actions = [action]
                  best_val = val
                elif val == best_val: # floating point comparison issues?
                  best_actions.append(action)
              chosen_action = src.random.choice(best_actions)
            
            # received_reward, terminal = env.step(action=chosen_action)
            observation, received_reward, terminated, truncated, info = env.step(action=chosen_action)
            all_rewards.append(received_reward)
            avg_rewards[step//s] += received_reward/s # NOTE probably wrong


            self.N[chosen_action] += 1
            q_val = self.Q[chosen_action]
            print(received_reward)
            self.Q[chosen_action] = q_val + (received_reward - q_val)/(self.N[chosen_action])
            # print(self.Q)

            if terminated: # NOTE uh almost certainly wrong
              env.reset()

        return np.vstack([self.Q for _ in range(n_states)]), avg_rewards

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the MultiArmedBandit algorithm and the state action values. Predictions
        are run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. During prediction, you
            should only "exploit."
          - You should use a loop to predict over each step in an episode until
            it terminates by returning `terminated or truncated=True`.
          - When choosing to exploit the best action, do not use np.argmax: it
            will deterministically break ties by choosing the lowest index of
            among the tied values. Instead, please *randomly choose* one of
            those tied-for-the-largest values.

        Arguments:
          env - (Env) An OpenAI Gymnasium environment with discrete actions and
            observations. See the OpenAI Gymnasium documentation for example use
            cases (https://gymnasium.farama.org/api/env/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        # reset environment before your first action
        env.reset()

        states, actions, rewards = [], [], []

        n_actions, n_states = env.action_space.n, env.observation_space.n
        try:
          self.Q
        except AttributeError: # super ratchet
          self.Q = np.zeros(n_actions)

        terminated = False
        while not terminated:

          best_val = -float('inf')
          for action, val in enumerate(state_action_values[0]): # NOTE this is hacky asf
            if val > best_val:
              best_actions = [action]
              best_val = val
            elif val == best_val: # floating point comparison issues?
              best_actions.append(action)
          chosen_action = src.random.choice(best_actions)

          observation, received_reward, terminated, truncated, info = env.step(action=chosen_action)

          actions.append(chosen_action)
          states.append(observation)
          rewards.append(received_reward)

        return states, actions, rewards
