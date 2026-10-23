---
layout: book-review
title: Reinforcement Learning - An Introduction
author: Richard S. Sutton and Andrew G. Barto
cover: assets/img/book_covers/rl.png
# olid: OL43499941M # use Open Library ID to fetch cover (if no `cover` is provided)
# isbn: 7539967447 # use ISBN to fetch cover (if no `olid` is provided, dashes are optional)
categories: reinforcement-learning
tags: top-100
# buy_link: https://www.amazon.com/Godfather-Deluxe-Mario-Puzo/dp/0593542592
# started: 2024-08-23
# finished: 2024-09-07
# released: 1969
stars: 5
# goodreads_review: 6318556633
status: finished
---

Reinforcement Learning: An Introduction, by Richard S. Sutton and Andrew G. Barto, is the foundational textbook of the field. The **second edition** (MIT Press, 2018) is a major expansion of the 1998 original, and the authors make it freely available online at [incompleteideas.net](http://incompleteideas.net/book/the-book-2nd.html). In March 2025, Sutton and Barto received the **2024 ACM A.M. Turing Award** "for developing the conceptual and algorithmic foundations of reinforcement learning", much of which this book sets out.

### What it covers

- **Part I, tabular solution methods:** multi-armed bandits and exploration vs. exploitation, finite Markov decision processes and Bellman equations, dynamic programming (policy and value iteration), Monte Carlo methods, temporal-difference learning (TD(0), SARSA, Q-learning, Expected SARSA), *n*-step bootstrapping, and planning and learning with models (Dyna, prioritized sweeping, Monte Carlo tree search).
- **Part II, approximate solution methods:** on-policy prediction and control with function approximation (including neural networks), off-policy methods and the "deadly triad", eligibility traces and TD(λ), and policy-gradient methods (REINFORCE, actor-critic).
- **Part III, looking deeper:** connections between RL and psychology (classical and instrumental conditioning) and neuroscience (dopamine and reward-prediction errors), plus case studies such as TD-Gammon, Watson's wagering strategy, Atari-playing DQN and AlphaGo/AlphaGo Zero.

### Why it's worth reading

Reinforcement learning has moved from the edge of AI to its center: it's behind game-playing systems like AlphaGo and is a core part of training today's large language models, through RLHF and RL for reasoning. Policy gradients, value functions, advantage estimation and the exploration–exploitation trade-off are the building blocks of algorithms such as PPO and GRPO, and this book explains them from first principles with clear intuition and pseudocode.
