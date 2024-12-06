## Question 2. Bandits vs. Q-Learning
a. `TODO`
b. `TODO`
c. `TODO`
d. `TODO`
e. `TODO`

## Question 3. Exploration vs. Exploitation
a. `Look at the code in q3.py and run python -m q3.py and include the plot it creates (free_response/3a_g0.9_a0.2.png) as your answer to this part. In your own words, describe what this code is doing.`
The code in q3.py implements Q-learning agent on the FrozenLake-v1 environment to analyze the impact of different values of epsilon (ε) (0.008, 0.08, and 0.8) (line 13) on learning performance. The environment, an 8x8 slippery grid (line 11), shows a reinforcement learning task where the agent must balance exploration and exploitation to maximize rewards. Exploration is trying new actions to potentially discover better rewards while exploitation is choosing actions that are currently believed to yield the best rewards. The agent is trained over 50,000 steps (line 16) for each epsilon value, with hyperparameters such as discount factor (𝛾 = 0.9) (line 17) and a learning rate (α = 0.2) (line 18) influencing how the agent updates its action-value estimates. 
For each epsilon, the code runs 10 trials (line 15) to ensure results are statistically robust, storing the overall rewards across training episodes. The agent uses the QLearing class (line 30), where the epsilon-greedy strategy determines whether it selects a random action (exploration) or the current best action (exploitation). The results are averaged across trials (line 39) and plotted to compare the performance of the different ε values. This code shows a clear visualization of how varying the ε affects the agent’s ability to balance exploration and exploitation, via the graph made, shown below, and saved as 3a_g0.9_a0.2.png in the free_response file. 

b. `Using the above plot, describe what you notice. What seems to be the ``best'' value of epsilon? What explains this result?`
The plot shows that the model’s performance varies significantly across the three ε values. An ε value of 0.08 results in the best overall performance, with the highest and most consistent rewards by the end of training. This happens because it finds a balance between exploration and exploitation, allowing the model to discover better strategies while effectively utilizing its knowledge. However, the ε value of 0.008 performs worse as it prioritizes exploitation too early, limiting its ability to seek optimal strategies. Finally, an ε value of 0.8 performs poorly because the high amount of exploration prevents the model from focusing on high-reward actions, resulting in slower performance growth and lower overall rewards.


c. `The above plot trains agents for 50,000 timesteps each. Suppose we instead trained them for 500,000 or 5,000,000 timesteps. How would you expect the trends to change or remain the same for each of the three values of epsilon? Give a one-sentence explanation for each value.`
If the models were trained for longer (500,000 or 5,000,000 timesteps), the trends for each epsilon value would be as follows:
* ε = 0.008: This model would likely continue to perform relatively well because it heavily exploits its learned strategies. However, its lack of initial exploration might mean it never finds the globally optimal strategy, restricting its potential performance.
* ε = 0.08: The performance of this model would likely remain the best among the three, as it balances exploration and exploitation even with extended training, allowing it to refine its strategy further. 
* ε = 0.8: This model would improve over time as it continues exploring, but its performance still would be behind the others due to the high amount of randomness in action selection. 


d. `When people use reinforcement learning in practice, it can be difficult to choose epsilon and other hyperparameters. Instead of trying three options like we did above, suppose we tried 30 or 300 different choices. What might be the danger of choosing epsilon this way if we wanted to use our agent in a new domain?`
Testing 30 or 300 different ε values might lead to overfitting the choice of the epsilon to the specific training setup as reinforcement learning algoirthms are highly senstiive to hyperparameters. When they are tuned too much within one envrionment, it can lead to overfitting in a new model, potentially leading to poor generalization. So, in a new environment, the model’s performance could decline because the hyperparameter tuning was too specific to the original environment. This overfitting risks finding an epsilon that works well only under certain conditions, limiting the flexibility of the model’s behavior. Also, the computational cost of testing a large range of values could become too much, especially for more intricate environments or longer training periods. A better approach might involve flexible techniques, such as gradually reducing epsilon over time, to handle a wider variety of situations.  

`References for Question 3`
1. https://ai-ml-analytics.com/reinforcement-learning-exploration-vs-exploitation-tradeoff/
2. https://mlu-explain.github.io/reinforcement-learning/
3. https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/
4. https://medium.com/nerd-for-tech/first-look-at-reinforcement-learning-67688f36413d
5. https://arxiv.org/pdf/1709.06560

## Question 4. Tic-Tac-Toe
a. `TODO`
b. `TODO`
c. `TODO`

## Question 5. Fair ML in the Real World

a. `TODO`
b. `TODO`
c. `TODO`
d. `TODO`
e. `TODO`
