---
layout: post
title: "Deep Q-Networks: How Deep Q-Learning Learned to Play Atari"
date: 2026-11-22 10:00:00 +0530
description: The DQN algorithm behind DeepMind's Atari results, line by line - from approximate Q-learning to the target network, experience replay, and the engineering details that made it stable.
tags: [RL]
toc:
  sidebar: left
related_posts: true
pseudocode: true
---

## Overview

In 2015 DeepMind published a single algorithm that learned to play 49 different Atari 2600 games from nothing but the screen pixels and the score, reaching human-level play on many of them. The same network architecture and the same hyperparameters were used for every game. That algorithm is the **Deep Q-Network (DQN)**.

DQN is not a new learning rule. It is ordinary Q-learning with a neural network as the function approximator, plus two stabilizers that turn an unstable combination into something that trains reliably:

1. **A target network:** a second, frozen copy of the Q-network that supplies the regression targets.
2. **Experience replay:** a large buffer of past transitions that the network learns from in random minibatches.

This post walks from approximate Q-learning to the full DQN algorithm, goes through it line by line, and explains why each piece is there.

**Prerequisites:** the idea of a Markov decision process (states, actions, rewards, discount γ), the Q-function, and gradient descent on a squared-error loss.

**A companion post** goes deep on one question this post only answers briefly: *why does DQN keep two Q-networks?* It is [here]({% post_url 2026-11-22-dqn-why-two-q-networks %}).

## 1. From tabular Q-learning to approximate Q-learning

The optimal action-value function $$Q^*(s,a)$$ is the expected discounted return from taking action a in state s and acting optimally afterwards. It satisfies the Bellman optimality equation:

$$
Q^*(s,a) = \mathbb{E}_{s'}\Big[\, r + \gamma \max_{a'} Q^*(s', a') \;\Big|\; s, a \Big]
$$

**Tabular Q-learning** keeps one number per (s, a) pair. After observing a transition (s, a, r, s′) it nudges the entry toward a one-step bootstrapped target:

$$
\text{target}(s') = r + \gamma \max_{a'} Q_k(s', a'), \qquad
Q_{k+1}(s,a) \leftarrow Q_k(s,a) + \alpha \big(\text{target}(s') - Q_k(s,a)\big)
$$

A table is hopeless for Atari. A state is a stack of 84×84 frames, so the number of distinct states is astronomically large and almost every state is seen at most once. The fix is to replace the table with a **parameterized function** $$Q_\theta(s,a)$$, for example a neural network with weights θ, so that what is learned about one state generalizes to similar ones.

**Approximate Q-learning** keeps the same idea: compute the bootstrapped target with the current parameters θ_k, then take a gradient step that moves the prediction toward it under squared error:

$$
\text{target}(s') = r + \gamma \max_{a'} Q_{\theta_k}(s', a')
$$

$$
\theta_{k+1} \leftarrow \theta_k - \alpha \,\nabla_\theta \Big[\tfrac{1}{2}\big(Q_\theta(s,a) - \text{target}(s')\big)^2\Big]_{\theta=\theta_k}
$$

The gradient is taken only through the prediction $$Q_\theta(s,a)$$. The target is treated as a fixed label even though it was computed from the same parameters. This is called a **semi-gradient** update. Expanding it:

$$
\theta_{k+1} = \theta_k + \alpha \big(\text{target}(s') - Q_{\theta_k}(s,a)\big)\, \nabla_\theta Q_{\theta}(s,a)\Big|_{\theta=\theta_k}
$$

For a table, $$\nabla_\theta Q_\theta(s,a)$$ is a one-hot vector at entry (s, a), and this reduces exactly to the tabular update above. For a neural network, the same error signal is spread across all the weights.

## 2. The DQN training algorithm

Here is the full algorithm from Mnih et al. (2015), "deep Q-learning with experience replay":

```pseudocode
\begin{algorithm}
\caption{Deep Q-learning with experience replay}
\begin{algorithmic}
\STATE Initialize replay memory $D$ to capacity $N$
\STATE Initialize action-value function $Q$ with random weights $\theta$
\STATE Initialize target action-value function $\hat{Q}$ with weights $\theta^{-} = \theta$
\FOR{episode $= 1$ \TO $M$}
    \STATE Initialize sequence $s_1 = \{ x_1 \}$ and preprocessed sequence $\phi_1 = \phi(s_1)$
    \FOR{$t = 1$ \TO $T$}
        \STATE With probability $\varepsilon$ select a random action $a_t$
        \STATE otherwise select $a_t = \arg\max_a Q(\phi(s_t), a; \theta)$
        \STATE Execute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$
        \STATE Set $s_{t+1} = s_t, a_t, x_{t+1}$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$
        \STATE Store transition $(\phi_t, a_t, r_t, \phi_{t+1})$ in $D$
        \STATE Sample random minibatch of transitions $(\phi_j, a_j, r_j, \phi_{j+1})$ from $D$
        \IF{episode terminates at step $j+1$}
            \STATE $y_j = r_j$
        \ELSE
            \STATE $y_j = r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^{-})$
        \ENDIF
        \STATE Perform a gradient descent step on $\left(y_j - Q(\phi_j, a_j; \theta)\right)^2$ with respect to the network parameters $\theta$
        \STATE Every $C$ steps reset $\hat{Q} = Q$
    \ENDFOR
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

Compare it with the approximate Q-learning update in section 1. The core is identical: a squared-error regression of $$Q(\phi_j, a_j; \theta)$$ toward a bootstrapped target $$y_j$$. Everything else in the algorithm is about *which* target and *which* data:

| Piece | Approximate Q-learning | DQN |
|---|---|---|
| Prediction (gets gradients) | $$Q_\theta(s,a)$$ | $$Q(\phi_j, a_j; \theta)$$, the **online network** |
| Target (treated as a fixed label) | $$Q_{\theta_k}(s', a')$$, always the latest weights | $$\hat{Q}(\phi_{j+1}, a'; \theta^-)$$, the **target network**, refreshed every C steps |
| Data | the transition just experienced | a random minibatch from a replay buffer |

The following sections take these differences one at a time.

## 3. One forward pass, one Q-value per action

The DQN network takes a state and outputs **one Q-value for every action** at once. For Atari that is up to 18 numbers from a single forward pass.

This design makes both maxes in the algorithm cheap:

- Acting needs $$\arg\max_a Q(\phi(s_t), a; \theta)$$: one forward pass, then pick the largest output.
- The target needs $$\max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^-)$$: one forward pass of the target network, then take the largest output.

The alternative, feeding (state, action) pairs into the network, would need 18 forward passes for every max.

During training, only the output for the action actually taken, $$a_j$$, receives an error signal. The other 17 outputs get zero gradient from that sample. They are trained by other samples in which those actions were taken.

## 4. Why DQN keeps two Q-networks (in brief)

The algorithm has two networks with the same architecture:

- **Online network** $$Q(\cdot\,; \theta)$$: trained by every gradient step, and used to pick actions.
- **Target network** $$\hat{Q}(\cdot\,; \theta^-)$$: never trained by gradient descent. It only computes the targets $$y_j$$, and it is overwritten with a copy of θ every C steps (C = 10,000 in the paper).

Both roles were already present in approximate Q-learning: the prediction $$Q_\theta(s,a)$$ and the target $$Q_{\theta_k}(s',a')$$. The only difference is that approximate Q-learning refreshes the target parameters after every single update, while DQN lets them **lag behind** for C steps.

**Why the lag matters, in one paragraph.** Q-learning has no dataset of correct answers: the network writes its own labels from its current guesses. In a table, updating Q(s, a) leaves Q(s′, ·) untouched, so the label stays put. In a neural network, updates are global. In Atari, s and s′ are consecutive frames that are nearly pixel-identical, so pushing Q(s, a) up also pushes Q(s′, ·) up, which raises the label, which pushes Q(s, a) up again. The network is chasing a target that moves every time it steps toward it. Freezing θ⁻ for C steps holds the labels still, so each window becomes an ordinary supervised regression problem with fixed labels, which gradient descent handles well.

**The deeper view:** each sync is roughly one step of value iteration, $$Q_{\text{new}} \approx \mathcal{T} Q_{\text{old}}$$, where $$\mathcal{T}$$ is the Bellman optimality backup. Between syncs, gradient descent solves "fit the Bellman backup of the old Q". DQN is approximately a sequence of supervised learning problems, each bootstrapped from the previous answer; this is the same idea as fitted Q-iteration.

**The trade-off:** a larger C gives more stability but slower propagation of new information, because the targets are computed from older knowledge.

The full story, with a worked numerical example of the moving-target problem, a step-by-step comparison with and without the frozen copy, what can still go wrong during the C frozen steps, and how Double DQN reuses the two networks, is in the companion post: **[Why DQN uses two Q-networks]({% post_url 2026-11-22-dqn-why-two-q-networks %})**.

## 5. The other half of DQN: experience replay

The target network fixes the *label* side of the regression. Experience replay fixes the *data* side.

Without replay, the agent would learn from transitions in the order they happen. That breaks things in three ways:

1. **Correlated samples.** Stochastic gradient descent assumes roughly independent samples, but 32 consecutive frames are essentially the same situation repeated. Their gradients all point the same way, so the network overfits to "right now".
2. **A feedback loop through the policy.** The current Q decides which actions are taken, which decides what data arrives next. If Q starts preferring "go left", the agent mostly sees left-side data, which reinforces left. The training distribution chases the policy.
3. **Forgetting and waste.** Each transition would be used once and thrown away, and what was learned on level 1 gets overwritten by level-2 data.

**The fix:** store every transition $$(\phi_t, a_t, r_t, \phi_{t+1})$$ in a large buffer D (1 million transitions in the paper) and train on **random minibatches** drawn from it. This:

- breaks the correlations, so a minibatch looks roughly i.i.d.;
- averages over many past behaviours, which smooths the data distribution;
- reuses each experience many times, which makes learning far more data-efficient.

**Why replay is allowed at all: Q-learning is off-policy.** The target $$r + \gamma \max_{a'} Q(s', a')$$ does not depend on which policy produced the transition. It evaluates the *greedy* policy no matter what behaviour generated the data, so learning from old transitions generated by an older, more exploratory policy is still valid. An on-policy method like SARSA, whose target uses the action the current policy would take, could not reuse old data this simply.

In DeepMind's ablations both stabilizers mattered, and removing replay generally hurt more than removing the target network.

## 6. Walking through the algorithm line by line

The algorithm is a loop between four components: the **environment** (the Atari emulator), the **agent** that acts in it, the **replay memory** that stores experience, and the **learner** that updates the network. Experience flows from agent to memory; sampled experience flows from memory to learner; gradients update the network; and updated parameters flow back to the agent, which uses them to choose actions.

### Setup

- **Replay memory D with capacity N:** a circular buffer. When it is full, the oldest transitions fall out.
- **Q with random weights θ, and $$\hat{Q}$$ with θ⁻ = θ:** the two networks start identical.

### Each time step: acting

1. **Preprocess with φ.** The raw frame sequence $$s_t = x_1, a_1, x_2, \ldots, a_{t-1}, x_t$$ is mapped by φ to a fixed-size input: each frame is converted to an 84×84 grayscale image, and the **last 4 frames are stacked**. A single frame cannot show velocity (which way the ball is moving), so stacking makes the input approximately Markov.
2. **ε-greedy action selection.** With probability ε pick a uniformly random action, otherwise pick $$\arg\max_a Q(\phi_t, a; \theta)$$. ε is annealed linearly from 1.0 to 0.1 over the first million frames, so the agent explores heavily early and exploits more later. Note that **acting uses the online network θ**, not the target network.
3. **Execute** the action in the emulator, observe the reward $$r_t$$ and the next frame $$x_{t+1}$$, preprocess, and **store** the transition $$(\phi_t, a_t, r_t, \phi_{t+1})$$ in D.

### Each time step: learning

4. **Sample a random minibatch** of transitions from D (32 in the paper).
5. **Compute targets with the frozen network:**

   $$
   y_j = \begin{cases} r_j & \text{if } s_{j+1} \text{ is terminal} \\ r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^-) & \text{otherwise} \end{cases}
   $$

   At a terminal state there is no future, so there is nothing to bootstrap from. Without the special case the agent would imagine value after "game over".
6. **Gradient step** on $$\big(y_j - Q(\phi_j, a_j; \theta)\big)^2$$ with respect to θ only. The gradient flows through the online network's output for $$a_j$$ alone, and $$y_j$$ is a constant: this is exactly the semi-gradient update from section 1.
7. **Every C steps, copy θ⁻ ← θ.**

### A minimal code sketch

The same loop in PyTorch-style Python, to make the data flow concrete:

```python
q_net = QNetwork(n_actions)                 # online network, weights theta
target_net = copy.deepcopy(q_net)           # target network, weights theta^- = theta
replay = ReplayBuffer(capacity=1_000_000)
opt = torch.optim.RMSprop(q_net.parameters(), lr=2.5e-4)

step = 0
for episode in range(M):
    phi = preprocess(env.reset())
    done = False
    while not done:
        # act with the ONLINE network (epsilon-greedy)
        if random.random() < epsilon(step):
            a = env.action_space.sample()
        else:
            a = q_net(phi[None]).argmax(dim=1).item()
        x_next, r, done = env.step(a)
        phi_next = preprocess(x_next)
        replay.add(phi, a, clip(r, -1, 1), phi_next, done)
        phi = phi_next

        # learn from a random minibatch
        s, a_b, r_b, s_next, done_b = replay.sample(32)
        with torch.no_grad():               # targets come from the FROZEN network
            max_next = target_net(s_next).max(dim=1).values
            y = r_b + gamma * max_next * (1 - done_b)
        q_sa = q_net(s).gather(1, a_b[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q_sa, y)    # Huber loss = error clipping
        opt.zero_grad(); loss.backward(); opt.step()

        step += 1
        if step % C == 0:                   # every C steps: theta^- <- theta
            target_net.load_state_dict(q_net.state_dict())
```

The `torch.no_grad()` block and the separate `target_net` are the two places where the algorithm's structure shows up in code: the target is a constant for the gradient, and it is computed from the lagging copy.

## 7. Practical details that made it work on Atari

- **Reward clipping to [−1, +1].** Games have wildly different score scales (one point per brick versus thousands per enemy). Clipping lets a single learning rate and a single set of hyperparameters work across all 49 games. The price is that the agent cannot tell a small reward from a large one.
- **Error clipping (Huber loss).** For large TD errors the loss is linear rather than quadratic, so its gradient is bounded. A single surprising transition cannot produce a huge update.
- **Frame skip.** The agent picks an action every 4th frame and repeats it on the frames in between. This makes training roughly four times cheaper with little loss in control.
- **The network.** Three convolutional layers (32 filters 8×8 stride 4, 64 filters 4×4 stride 2, 64 filters 3×3 stride 1), then a fully connected layer of 512 units, then one linear output per action. Inputs are pixels; nothing game-specific is hand-engineered.
- **Optimization.** RMSProp with learning rate 0.00025, minibatches of 32, discount γ = 0.99, replay memory of 1 million transitions, target update every C = 10,000 steps.

## 8. Why this was a breakthrough

Q-learning with neural networks had been tried before and was known to be unstable. The reason is what Sutton and Barto call the **deadly triad**: function approximation, bootstrapping and off-policy learning together can diverge. DQN kept all three, but added fixes for each concrete failure mode:

| Problem | Cause | DQN's fix |
|---|---|---|
| Moving targets | Bootstrapping + generalization | Target network θ⁻ |
| Correlated, non-stationary data | Learning online from sequential experience | Experience replay |
| Exploding gradients from scale differences | Varying reward scales, large TD errors | Reward clipping and Huber loss |

None of these is a convergence proof. They are engineering that makes the unstable combination work well in practice, and that was enough to reach human-level play on many Atari games from raw pixels.

## 9. One known flaw: overestimation, and Double DQN

The max in the target is **biased upward**. If the Q-estimates are noisy, $$\max_{a'}$$ picks whichever action happens to be overestimated, so $$\mathbb{E}[\max_{a'} \hat{Q}] \ge \max_{a'} \mathbb{E}[\hat{Q}]$$. That optimism then compounds through bootstrapping, because overestimated values become part of the next labels.

**Double DQN** (van Hasselt et al., 2016) fixes this with the two networks DQN already has. The online network *chooses* the next action and the target network *evaluates* it:

$$
y = r + \gamma\, \hat{Q}\big(s', \arg\max_{a'} Q(s', a'; \theta);\; \theta^-\big)
$$

Because the two networks' errors are partly independent, an action that the online network overestimates by chance is usually not also overestimated by the target network, and the bias largely cancels. It is a one-line change to the target computation.

## 10. The whole algorithm in 60 seconds

Q-learning learns how good each action is by regressing toward a target: the reward plus the best value one step later. DQN does this with a convolutional network that looks at game pixels and outputs a score for every action.

Doing this naively is unstable for two reasons. First, the target is computed by the same network that is being trained, so every update moves the target too, like a dog chasing its own tail. DQN fixes this with a frozen copy of the network that produces the targets and is refreshed only every few thousand steps. Second, consecutive game frames are almost identical, so learning from them in order is like studying one page of a textbook a thousand times. DQN stores experiences in a large memory and trains on random samples from it.

Frozen targets plus shuffled memories turn an unstable problem into something close to ordinary supervised learning, and that is what let one algorithm learn dozens of Atari games from raw pixels.

## Sources

- Mnih, V. et al. (2015). [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). *Nature* 518, 529–533. The DQN algorithm, architecture, hyperparameters and ablations.
- Mnih, V. et al. (2013). [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602). arXiv:1312.5602. The earlier version with experience replay but without a separate target network.
- van Hasselt, H., Guez, A. and Silver, D. (2016). [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461). AAAI. Double DQN.
- Sutton, R. S. and Barto, A. G. (2018). [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), 2nd ed., Chapter 11. The deadly triad.
