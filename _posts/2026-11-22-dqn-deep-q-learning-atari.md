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
Q^*(s,a) = \mathbb{E}_{s'}\Big[\, r + \gamma \max_{a'} Q^*(s', a') \;\Big\vert \; s, a \Big]
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
\theta_{k+1} = \theta_k + \alpha \big(\text{target}(s') - Q_{\theta_k}(s,a)\big)\, \nabla_\theta Q_{\theta}(s,a)\Big\vert _{\theta=\theta_k}
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
- **Target network** $$\hat{Q}(\cdot\,; \theta^-)$$: never trained by gradient descent. It only computes the targets $$y_j$$, and it is overwritten with a copy of θ every C steps (C = 10,000 gradient updates in the paper).

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

DQN samples uniformly from the buffer. Sampling the more informative transitions more often is the idea behind prioritized experience replay (section 11).

**Why replay is allowed at all: Q-learning is off-policy.** The target $$r + \gamma \max_{a'} Q(s', a')$$ does not depend on which policy produced the transition. It evaluates the *greedy* policy no matter what behaviour generated the data, so learning from old transitions generated by an older, more exploratory policy is still valid. An on-policy method like SARSA, whose target uses the action the current policy would take, could not reuse old data this simply.

DeepMind's ablation (Nature paper, Extended Data Table 3; 10 million training frames, best evaluation score) shows that both stabilizers matter:

| Game | Replay + target Q | Replay, no target Q | No replay, target Q | Neither |
|---|---|---|---|---|
| Breakout | 316.8 | 240.7 | 10.2 | 3.2 |
| Enduro | 1006.3 | 831.4 | 141.9 | 29.1 |
| River Raid | 7446.6 | 4102.8 | 2867.7 | 1453.0 |
| Seaquest | 2894.4 | 822.6 | 1003.0 | 275.8 |
| Space Invaders | 1088.9 | 826.3 | 373.2 | 302.0 |

Removing either one hurts in every game, and removing both is worst. In four of the five games, removing replay hurts more than removing the target network (Seaquest is the exception).

## 6. Walking through the algorithm line by line

The algorithm is a loop between four components: the **environment** (the Atari emulator), the **agent** that acts in it, the **replay memory** that stores experience, and the **learner** that updates the network. Experience flows from agent to memory; sampled experience flows from memory to learner; gradients update the network; and updated parameters flow back to the agent, which uses them to choose actions.

### Setup

- **Replay memory D with capacity N:** a circular buffer. When it is full, the oldest transitions fall out.
- **Q with random weights θ, and $$\hat{Q}$$ with θ⁻ = θ:** the two networks start identical.

### Each time step: acting

1. **Preprocess with φ.** The raw frame sequence $$s_t = x_1, a_1, x_2, \ldots, a_{t-1}, x_t$$ is mapped by φ to a fixed-size input. Each raw 210×160 colour frame is first combined with the previous frame by taking the per-pixel maximum (some Atari games draw certain objects only on alternate frames, so this removes flicker); then the luminance channel is extracted and rescaled to 84×84; and the **last 4 such frames are stacked**. A single frame cannot show velocity (which way the ball is moving), so stacking makes the input approximately Markov.
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

## 7. The Atari network architecture

The Q-network is a convolutional neural network with three design choices:

- **A history of frames as input.** The input is the preprocessed state φ: the last 4 frames, each 84×84 grayscale, stacked into a 4×84×84 tensor.
- **One output per action.** The final layer is a fully connected linear layer with one unit per legal action, and unit a outputs $$Q(s, a)$$, the expected discounted return for taking action a in state s (section 3).
- **No game-specific features.** The same architecture, trained from pixels and score alone, is used for every game.

<figure style="margin: 1.5rem 0;">
<svg viewBox="0 0 760 230" width="100%" role="img" aria-label="DQN architecture: a stack of 4 frames of 84 by 84 pixels feeds three convolutional layers of rectified linear units, then a fully connected layer of 512 rectified linear units, then a linear output layer with one Q-value per action." style="max-width: 760px; color: inherit;">
  <defs>
    <marker id="dqn-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
  </defs>
  <g fill="var(--global-theme-color)" fill-opacity="0.25" stroke="currentColor" stroke-width="1.2">
    <rect x="22" y="62" width="70" height="70"/><rect x="16" y="68" width="70" height="70"/><rect x="10" y="74" width="70" height="70"/><rect x="4" y="80" width="70" height="70"/>
    <rect x="150" y="66" width="44" height="44"/><rect x="146" y="70" width="44" height="44"/><rect x="142" y="74" width="44" height="44"/><rect x="138" y="78" width="44" height="44"/><rect x="134" y="82" width="44" height="44"/>
    <rect x="270" y="78" width="26" height="26"/><rect x="266" y="82" width="26" height="26"/><rect x="262" y="86" width="26" height="26"/><rect x="258" y="90" width="26" height="26"/><rect x="254" y="94" width="26" height="26"/>
    <rect x="378" y="86" width="18" height="18"/><rect x="374" y="90" width="18" height="18"/><rect x="370" y="94" width="18" height="18"/><rect x="366" y="98" width="18" height="18"/><rect x="362" y="102" width="18" height="18"/>
    <rect x="480" y="30" width="22" height="150"/>
    <rect x="610" y="65" width="22" height="80"/>
  </g>
  <g stroke="currentColor" stroke-width="1.2" fill="none" marker-end="url(#dqn-arrow)">
    <line x1="96" y1="110" x2="128" y2="110"/>
    <line x1="200" y1="100" x2="248" y2="100"/>
    <line x1="302" y1="100" x2="356" y2="105"/>
    <line x1="402" y1="105" x2="472" y2="105"/>
    <line x1="508" y1="105" x2="602" y2="105"/>
    <line x1="638" y1="105" x2="690" y2="105"/>
  </g>
  <g fill="currentColor" font-family="inherit" font-size="12" text-anchor="middle">
    <text x="48" y="48">4×84×84</text>
    <text x="45" y="172">stack of 4</text><text x="45" y="187">frames</text>
    <text x="165" y="52">32 · 8×8, s4</text>
    <text x="160" y="150">20×20×32</text><text x="160" y="165">conv + ReLU</text>
    <text x="276" y="62">64 · 4×4, s2</text>
    <text x="272" y="142">9×9×64</text><text x="272" y="157">conv + ReLU</text>
    <text x="382" y="72">64 · 3×3, s1</text>
    <text x="380" y="142">7×7×64</text><text x="380" y="157">conv + ReLU</text>
    <text x="491" y="20">512 units</text>
    <text x="491" y="200">fully connected</text><text x="491" y="215">+ ReLU</text>
    <text x="621" y="55">&#124;A&#124; outputs</text>
    <text x="621" y="165">linear</text><text x="621" y="180">output layer</text>
    <text x="725" y="101">Q(s, a)</text><text x="725" y="116">for each a</text>
  </g>
</svg>
<figcaption style="text-align: center; font-size: 0.9em;">The Nature (2015) DQN network. "32 · 8×8, s4" means 32 filters of size 8×8 with stride 4.</figcaption>
</figure>

The diagram shows the network from the Nature paper (2015), which produced the final results. The earlier arXiv paper (2013) used a smaller network with only **two** convolutional layers (16 filters of 8×8 with stride 4, then 32 filters of 4×4 with stride 2) and a fully connected layer of **256** rectifier units. The Nature version added a third convolutional layer, doubled the filters, and doubled the hidden layer to 512 units: "3 convolutional + 1 fully connected hidden layers". The two side by side:

| Layer | 2013 network (arXiv) | 2015 network (Nature, final results) | Output shape (2015) |
|---|---|---|---|
| Input | 4×84×84 | 4×84×84 | 4×84×84 |
| Conv 1 + ReLU | 16 filters, 8×8, stride 4 | 32 filters, 8×8, stride 4 | 32×20×20 |
| Conv 2 + ReLU | 32 filters, 4×4, stride 2 | 64 filters, 4×4, stride 2 | 64×9×9 |
| Conv 3 + ReLU | none | 64 filters, 3×3, stride 1 | 64×7×7 = 3,136 |
| Fully connected + ReLU | 256 units | 512 units | 512 |
| Linear output | one unit per action | one unit per action | \|A\| (up to 18) |
| Parameters (18 actions) | about 0.68 million | about 1.69 million | |

The spatial sizes follow from (input − filter) / stride + 1: (84 − 8)/4 + 1 = 20, then (20 − 4)/2 + 1 = 9, then (9 − 3)/1 + 1 = 7. Most of the parameters sit in the first fully connected layer (3,136 × 512 ≈ 1.6 million in the 2015 network). The number of actions varied between 4 and 18 across games; only the size of the output layer changes from game to game.

The two papers also preprocess slightly differently. The 2013 paper converted frames to grayscale, down-sampled them to 110×84, and then **cropped** an 84×84 region covering roughly the playing area. The 2015 paper instead takes the per-pixel max over two consecutive frames, extracts the luminance channel and **rescales** the whole frame to 84×84. Both stack the 4 most recent processed frames.

## 8. Training details that matter

Beyond the target network and replay, three choices in the optimization made a large practical difference.

### Huber loss instead of squared loss on the Bellman error

Write the Bellman (TD) error for one sample as $$a = y_j - Q(\phi_j, a_j; \theta)$$. Instead of the squared loss $$\tfrac{1}{2}a^2$$, DQN uses the **Huber loss** with threshold δ:

$$
L_\delta(a) = \begin{cases} \tfrac{1}{2} a^2 & \text{for } \lvert a\rvert  \le \delta, \\[4pt] \delta\big(\lvert a\rvert  - \tfrac{1}{2}\delta\big) & \text{otherwise.} \end{cases}
$$

It is quadratic near zero and linear in the tails, and the two pieces meet with the same value and the same slope at \|a\| = δ. Its derivative is what matters for training:

$$
\frac{\partial L_\delta}{\partial a} = \begin{cases} a & \text{for } \lvert a\rvert  \le \delta, \\ \delta \cdot \operatorname{sign}(a) & \text{otherwise,} \end{cases}
$$

so the gradient is the TD error **clipped to [−δ, δ]**. With δ = 1 this is exactly the "error clipping" of the Nature paper: the error term in the update is clipped to [−1, 1].

<figure style="margin: 1.5rem 0;">
<svg viewBox="0 0 540 300" width="100%" role="img" aria-label="Squared loss one half a squared and Huber loss with delta equal to 1, plotted for a from minus 4 to 4. They coincide between minus 1 and 1; outside that range the squared loss grows to 8 while the Huber loss grows linearly to 3.5." style="max-width: 540px; color: inherit;">
  <g stroke="currentColor" stroke-opacity="0.15" stroke-width="1">
    <line x1="50" y1="200" x2="500" y2="200"/><line x1="50" y1="140" x2="500" y2="140"/><line x1="50" y1="80" x2="500" y2="80"/>
    <line x1="106.25" y1="20" x2="106.25" y2="260"/><line x1="162.5" y1="20" x2="162.5" y2="260"/><line x1="218.75" y1="20" x2="218.75" y2="260"/><line x1="275" y1="20" x2="275" y2="260"/><line x1="331.25" y1="20" x2="331.25" y2="260"/><line x1="387.5" y1="20" x2="387.5" y2="260"/><line x1="443.75" y1="20" x2="443.75" y2="260"/>
  </g>
  <rect x="50" y="20" width="450" height="240" fill="none" stroke="currentColor" stroke-opacity="0.6"/>
  <path d="M50,20 Q275,500 500,20" fill="none" stroke="#4e79a7" stroke-width="2.5"/>
  <path d="M50,155 L218.75,245 Q275,275 331.25,245 L500,155" fill="none" stroke="#59a14f" stroke-width="2.5" stroke-dasharray="7 4"/>
  <g fill="currentColor" font-family="inherit" font-size="12" text-anchor="middle">
    <text x="50" y="278">−4</text><text x="106.25" y="278">−3</text><text x="162.5" y="278">−2</text><text x="218.75" y="278">−1</text><text x="275" y="278">0</text><text x="331.25" y="278">1</text><text x="387.5" y="278">2</text><text x="443.75" y="278">3</text><text x="500" y="278">4</text>
    <text x="275" y="296">TD error a</text>
  </g>
  <g fill="currentColor" font-family="inherit" font-size="12" text-anchor="end">
    <text x="44" y="264">0</text><text x="44" y="204">2</text><text x="44" y="144">4</text><text x="44" y="84">6</text><text x="44" y="24">8</text>
  </g>
  <g font-family="inherit" font-size="13">
    <line x1="300" y1="40" x2="326" y2="40" stroke="#4e79a7" stroke-width="2.5"/><text x="332" y="44" fill="currentColor">squared loss ½a²</text>
    <line x1="300" y1="62" x2="326" y2="62" stroke="#59a14f" stroke-width="2.5" stroke-dasharray="7 4"/><text x="332" y="66" fill="currentColor">Huber loss, δ = 1</text>
  </g>
</svg>
<figcaption style="text-align: center; font-size: 0.9em;">Squared loss versus Huber loss (δ = 1). They are identical for &#124;a&#124; ≤ 1; beyond that the Huber loss grows only linearly.</figcaption>
</figure>

**Why this matters more in RL than in supervised learning.** In Q-learning the labels are bootstrapped from the network's own estimates, so early in training, after a target-network copy, or on a rare transition with a surprising reward, the TD error can be very large. Under squared loss the gradient grows in proportion to the error, so a single such sample can produce a huge update that damages the values of many other states, and those damaged values then flow into future labels. The Huber loss bounds each sample's influence: a transition with TD error 10 pushes no harder than one with TD error 1. It also makes the loss less sensitive to outliers, behaving like an absolute-error loss far from zero.

Note that this is different from clipping the loss value or the total gradient norm. Only the per-sample error term in the gradient is clipped; the direction $$\nabla_\theta Q(\phi_j, a_j; \theta)$$ is unchanged. In PyTorch this is `F.smooth_l1_loss` (or `F.huber_loss` with `delta=1.0`), as in the code sketch above.

### RMSProp instead of vanilla SGD

Plain stochastic gradient descent uses one global step size for every parameter. DQN instead uses **RMSProp**, which divides each parameter's gradient by a running estimate of its recent magnitude:

$$
v \leftarrow \rho\, v + (1-\rho)\, g^2, \qquad
\theta \leftarrow \theta - \frac{\alpha}{\sqrt{v + \epsilon}}\, g
$$

where g is the minibatch gradient and the operations are element-wise. The Nature paper lists a learning rate α = 0.00025, a "squared gradient momentum" ρ = 0.95, a "min squared gradient" of 0.01 added in the denominator, and also a "gradient momentum" of 0.95: a running average of g itself, which DeepMind's released code uses to centre the denominator ($$\sqrt{v - m^2 + \epsilon}$$ with $$m$$ the running mean of g), a variant often called centered RMSProp. The 2013 paper says only that RMSProp was used with minibatches of 32.

The point is that **optimization in RL really matters**, more than the RL-specific parts of the algorithm might suggest:

- **The regression target keeps changing.** Every target-network copy changes the labels, and as the values grow from near zero toward their true scale, the size of the gradients changes too. A fixed SGD step size tuned for one phase is too large or too small in another. RMSProp adapts the per-parameter scale automatically.
- **Gradients are very uneven across parameters.** Convolutional filters in early layers, the large fully connected layer, and individual action outputs (which only receive gradient when their action was taken) see very different gradient magnitudes. Normalizing per parameter keeps rarely-updated outputs from learning too slowly and frequently-updated ones from overshooting.
- **Bad steps are expensive.** In supervised learning a bad step costs some loss that later steps repair. In Q-learning a bad step corrupts values that become future labels and change the behaviour policy, so the damage propagates. A well-conditioned optimizer reduces how often that happens.

Later DQN variants (for example Rainbow) switched to **Adam**, which adds momentum to the same per-parameter normalization, and found it less sensitive to the learning rate.

### Annealing the exploration rate

With ε-greedy action selection, ε is the probability of taking a uniformly random action. It helps to **anneal** it rather than keep it fixed: start ε at 1 and decrease it linearly to 0.1 (or 0.05) over the first million frames, then hold it there:

$$
\varepsilon(t) = \max\Big(\varepsilon_{\text{final}},\; 1 - (1 - \varepsilon_{\text{final}})\,\frac{t}{10^6}\Big), \qquad \varepsilon_{\text{final}} \in \{0.1,\, 0.05\}
$$

The reasoning:

- **At the start, the Q-network is random**, so its greedy action is meaningless. Acting fully at random (ε = 1) fills the replay buffer with diverse experience and covers the state space as widely as possible. The paper also fills the buffer with 50,000 frames of random play before learning starts.
- **As the Q-values become meaningful**, acting greedily most of the time lets the agent reach later, more interesting parts of the game (a random agent rarely clears the first level), which is where the remaining learning signal is.
- **ε never goes to zero during training**, so the agent keeps trying alternatives and the data in the buffer never collapses onto a single trajectory. Because Q-learning is off-policy, these exploratory actions do not bias the target.

At evaluation time a small ε (0.05 in the Nature paper) is still used, which prevents the agent from getting stuck in a deterministic loop and avoids overfitting to one exact sequence of states.

### Other practical details

- **Reward clipping to [−1, +1].** Games have wildly different score scales (one point per brick versus thousands per enemy). Clipping lets one learning rate and one set of hyperparameters work across all 49 games. The price is that the agent cannot tell a small reward from a large one.
- **Frame skip.** The agent picks an action every 4th frame and repeats it on the frames in between. This makes training roughly four times cheaper with little loss in control.
- **Hyperparameters (Nature paper, Extended Data Table 1).** Minibatches of 32; discount γ = 0.99; replay memory of the 1 million most recent frames; one gradient update every 4 actions; target network update every C = 10,000 **gradient updates** (so roughly every 40,000 actions, or 160,000 emulator frames with action repeat 4); 50,000 frames of random play before learning starts; up to 30 random "no-op" actions at the start of each episode to vary the starting state; and training on 50 million frames per game. The 2013 paper trained for 10 million frames.

## 9. Why this was a breakthrough

Q-learning with neural networks had been tried before and was known to be unstable. The reason is what Sutton and Barto call the **deadly triad**: function approximation, bootstrapping and off-policy learning together can diverge. DQN kept all three, but added fixes for each concrete failure mode:

| Problem | Cause | DQN's fix |
|---|---|---|
| Moving targets | Bootstrapping + generalization | Target network θ⁻ |
| Correlated, non-stationary data | Learning online from sequential experience | Experience replay |
| Exploding gradients from scale differences | Varying reward scales, large TD errors | Reward clipping and Huber loss |

None of these is a convergence proof. They are engineering that makes the unstable combination work well in practice, and that was enough to reach human-level play on many Atari games from raw pixels.

## 10. One known flaw: overestimation, and Double DQN

Everything above is standard DQN. This section covers its best-known flaw and the smallest fix for it, **Double DQN** (van Hasselt, Guez and Silver, 2016).

### The upward bias in max

The DQN target contains $$\max_{a'} \hat{Q}(s', a'; \theta^-)$$. The network's estimates are noisy, and the max of noisy estimates is biased upward:

$$
\mathbb{E}\Big[\max_{a'} \hat{Q}(s', a')\Big] \;\ge\; \max_{a'} \mathbb{E}\Big[\hat{Q}(s', a')\Big]
$$

The reason is that the max uses the **same** numbers twice: once to *select* the action that looks best, and again to *evaluate* how good it is. The action that looks best is disproportionately likely to be one whose estimate happens to be too high, so the value reported for it is too high too.

A concrete case. In some next state s′ there are 4 actions, all with true value 1.0. The two networks' current estimates are:

| Action | True value | Online Q(s′, a; θ) | Target $$\hat{Q}$$(s′, a; θ⁻) |
|---|---|---|---|
| a₁ | 1.0 | **1.3** | 0.9 |
| a₂ | 1.0 | 0.8 | **1.2** |
| a₃ | 1.0 | 1.1 | 1.0 |
| a₄ | 1.0 | 0.9 | 1.1 |

The DQN target uses $$\max_{a'} \hat{Q}(s', a'; \theta^-) = 1.2$$, an overestimate of 0.2. Every estimate was off by a modest, zero-mean amount, but the max picked out the most positive error.

**How big is the bias?** If the errors of m actions are independent and uniform on [−1, 1], the expected max error is

$$
\mathbb{E}\Big[\max_a \epsilon_a\Big] = \frac{m-1}{m+1},
$$

which is 1/3 for 2 actions and 17/19 ≈ 0.89 for the 18 Atari actions (this is Theorem 2 in the Double DQN paper). Typically the overestimation **grows with the number of actions**, and it is far from negligible next to the error scale itself.

The bias is not stuck in one target. Each label is computed from values that were themselves learned from earlier optimistic labels, so the optimism **compounds through bootstrapping**. Roughly, if every backup adds a bias b, the values can drift up by as much as b + γb + γ²b + … = b/(1 − γ), which is 100b for γ = 0.99.

### Overestimation happens even if the errors are unbiased on average

One might hope the problem goes away when the estimates are correct on average. It does not. The paper proves a lower bound that holds for **any** source of error (noise, function approximation, non-stationarity), without assuming the errors are independent:

**Theorem 1 (van Hasselt et al., 2016).** In a state s where all true action values are equal, $$Q_*(s,a) = V_*(s)$$, let the estimates $$Q_t(s,a)$$ be unbiased on the whole, $$\sum_a \big(Q_t(s,a) - V_*(s)\big) = 0$$, but not all correct, with mean squared error $$\frac{1}{m}\sum_a \big(Q_t(s,a) - V_*(s)\big)^2 = C > 0$$, where m ≥ 2 is the number of actions. Then

$$
\max_a Q_t(s,a) \;\ge\; V_*(s) + \sqrt{\frac{C}{m-1}}.
$$

The bound is tight, and under the same conditions the lower bound on the absolute error of the Double Q-learning estimate is zero.

**A short proof.** Write $$\epsilon_a = Q_t(s,a) - V_*(s)$$ and $$M = \max_a \epsilon_a$$. Since the errors sum to zero and are not all zero, some are negative, so let n ≤ m − 1 be the number of non-negative errors. Their sum P is at most nM, and the negative errors must sum to −P. Then

- the non-negative errors contribute at most nM² to the sum of squares (each is at most M);
- the negative errors contribute at most P² ≤ n²M² (a sum of squares is at most the square of the sum of absolute values).

So $$mC = \sum_a \epsilon_a^2 \le nM^2 + n^2M^2 = n(n+1)M^2 \le (m-1)m\,M^2$$, which gives $$M \ge \sqrt{C/(m-1)}$$. The bound is attained by $$\epsilon_a = \sqrt{C/(m-1)}$$ for m − 1 actions and $$\epsilon_1 = -\sqrt{(m-1)C}$$ for the remaining one. ∎

The takeaway: zero-mean errors are not enough. As long as there is *any* error, the max is optimistic.

### Decouple selection from evaluation

The Q-learning target can be rewritten to show the two jobs the max does:

$$
\max_{a'} Q(s', a'; \theta) \;=\; Q\big(s', \underbrace{\arg\max_{a'} Q(s', a'; \theta)}_{\text{selection}};\; \underbrace{\theta}_{\text{evaluation}}\big)
$$

**Double Q-learning** (van Hasselt, 2010) breaks the link by learning two value functions with weights θ and θ′, assigning each experience at random to update one of them. One set selects the greedy action, the other evaluates it:

$$
y^{\text{DoubleQ}} = r + \gamma\, Q\big(s', \arg\max_{a'} Q(s', a'; \theta);\; \theta'\big)
$$

If the evaluator's errors are independent of the selector's, the action picked for having a lucky high error under θ gets an unbiased value under θ′, so the bias disappears on average. In the uniform-noise example above, the expected error becomes 0 instead of (m − 1)/(m + 1).

### Double DQN: reuse the target network as the second estimator

DQN already has a second set of weights, θ⁻. Double DQN uses it as the evaluator instead of training a separate network:

- **θ (online) selects** the best next action;
- **θ⁻ (target) evaluates** that action.

$$
y^{\text{DoubleDQN}} = r + \gamma\, \hat{Q}\big(s', \arg\max_{a'} Q(s', a'; \theta);\; \theta^-\big)
$$

and the loss at iteration i is

$$
L_i(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim D}\Big[\Big(r + \gamma\, Q\big(s', \arg\max_{a'} Q(s', a'; \theta_i);\; \theta_i^-\big) - Q(s, a; \theta_i)\Big)^2\Big]
$$

(with the Huber loss in place of the square in practice). Back to the 4-action table: the online network selects a₁ (its estimate, 1.3, is highest), and the target network evaluates a₁ at 0.9. The Double DQN target uses 0.9 instead of DQN's 1.2. In this single instance it happens to be 0.1 too low, but because the target network's error on a₁ is not tied to the online network's lucky +0.3, the errors average out to roughly zero over many targets instead of always being positive.

**Which weights do what**, compared with DQN:

| | Selects a′ | Evaluates a′ | Frozen? | Gradient flows? |
|---|---|---|---|---|
| DQN target | θ⁻ | θ⁻ | yes | no |
| Double DQN target | θ (online) | θ⁻ | θ: no, θ⁻: yes | no |
| Prediction Q(s, a; θ) | | | no | yes |

The online weights appear inside the Double DQN target, but only to pick an index; the target is still treated as a constant, and no gradient flows through the argmax.

**In the algorithm**, the only line that changes is the target:

```python
with torch.no_grad():
    # DQN:        max_next = target_net(s_next).max(dim=1).values
    a_star   = q_net(s_next).argmax(dim=1, keepdim=True)          # online network SELECTS
    max_next = target_net(s_next).gather(1, a_star).squeeze(1)    # target network EVALUATES
    y = r_b + gamma * max_next * (1 - done_b)
```

The cost is one extra forward pass of the online network on the next states. The target network is still a periodic copy of the online network, the replay buffer and everything else are unchanged. The paper calls this "perhaps the minimal possible change to DQN towards Double Q-learning".

**It is not a full decoupling.** θ⁻ is an old copy of θ, so the two networks' errors are correlated. Right after each copy they are identical, and the Double DQN target reduces to the DQN target until the online network moves away again. This is why the tuned version below copies less often.

### What it changes in practice

**Value estimates.** The paper trained DQN and Double DQN on Atari with 6 random seeds each, using DQN's hyperparameters, and compared the networks' value estimates with the true discounted return of the final policy (obtained by running the learned policy and averaging the actual discounted return from each visited state). Without bias the two would match.

- DQN's estimates consistently end up **far above** the true values. Overestimation was observed in all 49 games tested, in varying amounts.
- Double DQN's estimates are **much closer** to the true values. Its true values are also often higher than DQN's, so it learns better policies, not just better-calibrated numbers.
- On **Wizard of Wor** and **Asterix**, DQN's value estimates blow up (they need a log scale to plot), and the moment they do, the game score **drops**. Double DQN's estimates and scores stay stable. This suggests that much of DQN's instability in those games, which one might attribute to the deadly triad, is caused by Q-learning's overoptimism.

**Scores.** Scores are normalized per game so that 0% is a random agent and 100% is a professional human tester:

$$
\text{score}_{\text{normalized}} = \frac{\text{score}_{\text{agent}} - \text{score}_{\text{random}}}{\text{score}_{\text{human}} - \text{score}_{\text{random}}}
$$

Over the 49 games:

| | No-op starts: DQN | No-op starts: Double DQN | Human starts: DQN | Human starts: Double DQN | Human starts: Double DQN (tuned) |
|---|---|---|---|---|---|
| **Median** | 93.5% | 114.7% | 47.5% | 88.4% | 116.7% |
| **Mean** | 241.1% | 330.3% | 122.0% | 273.1% | 475.2% |

- **No-op starts:** each evaluation episode begins with up to 30 "do nothing" actions to vary the starting state, then 5 minutes of play with ε = 0.05, averaged over 100 episodes. Double DQN used **exactly the same hyperparameters as DQN**, which were tuned for DQN, so this comparison is if anything unfavourable to Double DQN. Large gains include Road Runner (233% → 617%), Asterix (70% → 180%), Zaxxon (54% → 111%) and Double Dunk (17% → 397%).
- **Human starts:** in a deterministic game with a single start, an agent could memorize one action sequence. To test generalization, 100 starting points per game are sampled from a human expert's trajectory and the agent plays from each for up to 30 minutes. DQN's median drops to 47.5% here, while Double DQN's nearly doubles it.
- **Double DQN (tuned):** three changes to the hyperparameters. (1) The target network is copied every **30,000** frames instead of 10,000, to reduce overestimation further, since right after each copy Double DQN reverts to ordinary Q-learning. (2) Exploration during learning is reduced from ε = 0.1 to **ε = 0.01**, with **ε = 0.001** at evaluation. (3) The top layer uses a **single shared bias** for all action values.

### When overestimation matters, and when it does not

Overestimation is not always harmful. If every action's value were overestimated by the **same** amount, the argmax, and therefore the policy, would not change. In Pong, DQN reaches optimal play despite slightly overestimating its values. The damage comes from **uneven** overestimation: some actions or states get inflated more than others because of where the noise and the data happen to be, the greedy policy shifts toward them, and bootstrapping spreads the inflated values to their predecessors. Reducing the bias is mostly about stability, and Double DQN delivers that at almost no cost.

The double estimator is not a free lunch either: with independent errors it can **underestimate** (as the single 0.9 instance above did), and in the 2010 analysis it is shown to be biased downward in some cases. In practice a mild pessimism is far less harmful than compounding optimism. Later actor-critic methods push the same idea further: TD3 takes the **minimum** of two critics' estimates in its target ("clipped double Q-learning"), and SAC does the same.

## 11. Prioritized experience replay

Experience replay (section 5) samples every stored transition with **equal probability**. That is simple, but it is highly suboptimal: most transitions in a large buffer are already well predicted and teach almost nothing, while a few rare ones (the first time a reward was found, a surprising death) carry most of the learning signal. **Prioritized experience replay** (Schaul, Quan, Antonoglou and Silver, 2016) replays transitions more often when there is more to learn from them.

### Why the order of replay matters: the Blind Cliffwalk

The paper motivates this with a small environment. There are n states in a row and two actions in each. The "right" action moves one state forward; the "wrong" action ends the episode. Only reaching the end gives a reward of 1, and everything else gives 0. A random agent reaches the reward with probability 2⁻ⁿ, so the single successful transition is buried among an exponential number of redundant failures, much like a robot that falls over thousands of times before it first walks.

With uniform replay, the agent wastes almost all its updates on failure transitions it has already learned. An oracle that always replays the transition that most reduces the global loss learns **exponentially faster** in n. The oracle is not realistic, but the gap shows how much there is to gain from replaying the right transitions.

### Prioritizing by TD error

The ideal criterion, how much the agent can learn from a transition, is not directly measurable. A practical proxy is the **magnitude of the TD error**, the absolute Bellman error, which says how surprising a transition is:

$$
\lvert \delta_i\rvert  = \Big\vert  r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \Big\vert 
$$

(when combined with Double DQN, as in the paper's experiments, the Double DQN target from section 10 is used inside the absolute value). Q-learning already computes δ for every update, so the priority costs nothing extra.

The simplest version, **greedy TD-error prioritization**, stores the last TD error with each transition and always replays the one with the largest \|δ\|. New transitions have no TD error yet, so they are inserted with the **maximum priority** seen so far, which guarantees every transition is replayed at least once. On the Blind Cliffwalk this already cuts the learning effort dramatically.

But greedy prioritization has three problems:

1. **Stale priorities.** To avoid sweeping the whole buffer, a TD error is only updated when its transition is replayed. A transition with a small error on its first visit may then never be replayed again before it drops out of the sliding-window buffer.
2. **Sensitivity to noise.** With stochastic rewards, or approximation errors amplified by bootstrapping, noise spikes produce large \|δ\| for transitions that have nothing to teach.
3. **Lack of diversity.** Errors shrink slowly under function approximation, so the same small set of high-error transitions is replayed over and over and the network overfits to them.

### Stochastic prioritization

The fix is to sample **stochastically**, with probability increasing in priority but never zero:

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}
$$

The exponent α controls how much prioritization is used: α = 0 is uniform sampling, and larger α approaches greedy prioritization. Two ways to define the priority $$p_i$$:

- **Proportional:** $$p_i = \lvert \delta_i\rvert  + \epsilon$$, where the small constant ϵ keeps a transition with zero error from never being revisited.
- **Rank-based:** $$p_i = 1/\text{rank}(i)$$, where rank(i) is the transition's position when the buffer is sorted by \|δ\|. P then follows a power law with exponent α. Because only the ordering matters, this variant is insensitive to outliers in the size of \|δ\|.

**A small example.** Four transitions have TD errors \|δ\| = 2, 1, 0.5 and 0.5, and α = 0.6 (proportional, ignoring ϵ). Then $$p^{\alpha}$$ = 1.516, 1.0, 0.660, 0.660, which sum to 3.835, giving sampling probabilities

| Transition | \|δ\| | Uniform P | Prioritized P |
|---|---|---|---|
| 1 | 2.0 | 0.25 | 0.395 |
| 2 | 1.0 | 0.25 | 0.261 |
| 3 | 0.5 | 0.25 | 0.172 |
| 4 | 0.5 | 0.25 | 0.172 |

The highest-error transition is replayed about 2.3 times as often as each low-error one. Without the α < 1 damping (α = 1) the ratio would be 4.

**Efficient sampling.** A buffer of a million transitions cannot be scanned on every sample.

- For the proportional variant, the paper uses a **sum-tree**: a binary tree whose leaves hold the priorities and whose internal nodes hold the sum of their children. The root is the total priority. To sample, draw a number uniformly in [0, total) and walk down the tree, going left or right depending on the left child's sum. Sampling and updating a priority both take O(log N).
- For the rank-based variant, the cumulative distribution is approximated piecewise with k segments of equal probability, with k equal to the minibatch size, and one transition is sampled from each segment. This is stratified sampling, so every minibatch contains a mix of high-, medium- and low-error transitions.

### Correcting the bias with importance sampling

Stochastic gradient updates estimate an expectation over the data distribution. Prioritized sampling changes that distribution, so without correction the values converge to a different, biased solution. The fix is to weight each sampled transition's update by an **importance-sampling (IS) weight**:

$$
w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^{\beta}
$$

normalized by $$1/\max_j w_j$$ so that weights only ever scale updates down. The update uses $$w_i \delta_i$$ in place of $$\delta_i$$. With β = 1 the non-uniform sampling is fully compensated.

In the four-transition example, with β = 1 the normalized weights are 0.435, 0.660, 1.0 and 1.0. The frequently sampled high-error transition gets a smaller step each time it is replayed, so on average its influence matches what uniform sampling would give. With β = 0.4 the weights are 0.717, 0.847, 1.0 and 1.0, a partial correction.

**Annealing β.** Unbiased updates matter most near convergence. Early in training the process is highly non-stationary anyway (the policy, the state distribution and the targets all keep changing), so a little bias is acceptable. The paper therefore **anneals β linearly from β₀ to 1** over training.

The IS weights have a second benefit. High-error transitions are seen many times, but each time with a reduced step, so the effective step size in parameter space stays small. This matters for deep networks, where large steps are disruptive because the gradient is only a local approximation.

### The algorithm

The paper's Algorithm 1 combines Double DQN with proportional prioritization:

```pseudocode
\begin{algorithm}
\caption{Double DQN with proportional prioritization}
\begin{algorithmic}
\REQUIRE minibatch size $k$, step-size $\eta$, replay period $K$ and size $N$, exponents $\alpha$ and $\beta$, budget $T$
\STATE Initialize replay memory $H = \emptyset$, $\Delta = 0$, $p_1 = 1$
\STATE Observe $S_0$ and choose $A_0 \sim \pi_\theta(S_0)$
\FOR{$t = 1$ \TO $T$}
    \STATE Observe $S_t, R_t, \gamma_t$
    \STATE Store transition $(S_{t-1}, A_{t-1}, R_t, \gamma_t, S_t)$ in $H$ with maximal priority $p_t = \max_{i<t} p_i$
    \IF{$t \equiv 0 \bmod K$}
        \FOR{$j = 1$ \TO $k$}
            \STATE Sample transition $j \sim P(j) = p_j^\alpha / \sum_i p_i^\alpha$
            \STATE Compute importance-sampling weight $w_j = (N \cdot P(j))^{-\beta} / \max_i w_i$
            \STATE Compute TD-error $\delta_j = R_j + \gamma_j Q_{\text{target}}(S_j, \arg\max_a Q(S_j, a)) - Q(S_{j-1}, A_{j-1})$
            \STATE Update transition priority $p_j \leftarrow |\delta_j|$
            \STATE Accumulate weight-change $\Delta \leftarrow \Delta + w_j \cdot \delta_j \cdot \nabla_\theta Q(S_{j-1}, A_{j-1})$
        \ENDFOR
        \STATE Update weights $\theta \leftarrow \theta + \eta \cdot \Delta$, reset $\Delta = 0$
        \STATE From time to time copy weights into target network $\theta_{\text{target}} \leftarrow \theta$
    \ENDIF
    \STATE Choose action $A_t \sim \pi_\theta(S_t)$
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

Compared with section 2, only three steps are new: sampling by priority, computing the IS weight, and writing the new \|δ\| back as the transition's priority. A minimal sum-tree in Python:

```python
class SumTree:
    """Leaves hold priorities p_i^alpha; each internal node holds the sum of its children."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)        # tree[1] is the root; leaves at [capacity, 2*capacity)

    def update(self, idx, priority):              # O(log N)
        i = idx + self.capacity
        self.tree[i] = priority
        while i > 1:
            i //= 2
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def total(self):
        return self.tree[1]

    def sample(self, u):                          # u ~ Uniform[0, total), O(log N)
        i = 1
        while i < self.capacity:
            left = 2 * i
            if u < self.tree[left]:
                i = left
            else:
                u -= self.tree[left]
                i = left + 1
        return i - self.capacity                  # index of the sampled transition

# Sampling a minibatch with stratification and IS weights:
#   segment = tree.total() / k
#   idx_j   = tree.sample(random.uniform(j * segment, (j + 1) * segment))   for j in range(k)
#   P_j     = tree.tree[idx_j + capacity] / tree.total()
#   w_j     = (N * P_j) ** (-beta);  w = w / w.max()
#   loss    = (w * huber(td_errors)).mean();  then tree.update(idx_j, (abs(td_j) + eps) ** alpha)
```

### Results

The experiments kept everything else identical to the baselines: same network, same learning algorithm, same buffer of 10⁶ transitions, minibatches of 32 with one update per 4 new transitions (so each transition is replayed 8 times on average under uniform sampling), and the same human-starts evaluation. Only two things changed:

- **The step size was reduced by a factor of 4**, because prioritized minibatches contain more high-error transitions and so produce larger gradients.
- **α and β₀** were set by a coarse grid search on 8 games: α = 0.7, β₀ = 0.5 for rank-based and α = 0.6, β₀ = 0.4 for proportional.

Normalized scores with human starts:

| | DQN baseline | DQN rank-based | Double DQN (tuned) baseline | Double DQN rank-based | Double DQN proportional |
|---|---|---|---|---|---|
| **Median** | 48% | 106% | 111% | 113% | 128% |
| **Mean** | 122% | 355% | 418% | 454% | 551% |
| **Games > baseline** | – | 41 | – | 38 | 42 |
| **Games > human** | 15 | 25 | 30 | 33 | 33 |
| **Number of games** | 49 | 49 | 57 | 57 | 57 |

- Adding prioritized replay to DQN improves the score substantially on **41 of 49 games**, and the median more than doubles, from 48% to 106%.
- The gain is **complementary to Double DQN**: on top of tuned Double DQN, proportional prioritization raises the median from 111% to 128% and the mean from 418% to 551%, bringing River Raid, Seaquest and Surround to human level for the first time. The mean is a fragile summary, since a single game (Video Pinball) dominates it.
- On aggregate, **learning is about twice as fast**: the prioritized agents reach the uniform-replay baseline's final median score in roughly half the training steps.
- Rank-based and proportional usually perform similarly, but on some games one stays near the baseline while the other gives a large boost.

## 12. Beyond DQN: the main extensions and Rainbow

DQN started a line of research in which each paper fixed one specific weakness. Double DQN (section 10) fixes overestimation and prioritized replay (section 11) fixes inefficient sampling. Four other extensions became standard, and the **Rainbow** agent combined all six. This section covers each one's core idea briefly.

### Dueling networks: separate the value of a state from the advantage of an action

**Paper:** Wang et al. (2016), *Dueling Network Architectures for Deep Reinforcement Learning*.

**The problem.** In many states the choice of action barely matters. In the racing game Enduro, steering left or right is only important when a collision is imminent; the rest of the time, every action leads to about the same outcome. Yet a standard Q-network must learn the value of each action separately, and each sample only updates the output of the one action taken. Meanwhile, accurate state values matter in **every** state for bootstrapping.

**The idea.** Split the network after the convolutional layers into two streams of fully connected layers. One outputs a scalar **state value** V(s); the other outputs an **advantage** A(s, a) for every action, where $$A(s,a) = Q(s,a) - V(s)$$. They are recombined into Q-values:

$$
Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \Big( A(s, a; \theta, \alpha) - \frac{1}{\lvert \mathcal{A}\rvert } \sum_{a'} A(s, a'; \theta, \alpha) \Big)
$$

Here θ are the shared convolutional parameters and α, β are the parameters of the two streams.

**Why subtract the mean.** The naive sum Q = V + A is **unidentifiable**: adding a constant to V and subtracting it from A gives the same Q, so nothing forces V to learn the state value. Forcing the advantages to have zero mean pins this down. (Subtracting the max instead would force the best action's advantage to be 0, which matches the definition of A for the greedy policy exactly. The mean version is more stable, because the advantages only need to change as fast as their mean, rather than compensating for every change in the best action's advantage.) Subtracting a constant does not change the ranking of actions, so acting greedily on Q is the same as acting greedily on A.

**Why it helps.** Every update to any action updates V, the part shared by all actions. The value of a state is learned from every sample, and the advantage stream only has to learn the differences between actions. In a corridor environment where extra no-op actions could be added, both architectures learned equally fast with 5 actions, but the dueling network pulled ahead as the action count rose to 10 and 20. Saliency maps on Enduro show the value stream watching the road and the score, while the advantage stream only attends to cars right in front of the agent.

**It is only an architecture change.** The output is still a vector of Q-values, so it works with any Q-learning algorithm (the paper trains it with Double DQN). Implementation details: each stream has a 512-unit fully connected layer, the gradient entering the last convolutional layer is rescaled by $$1/\sqrt{2}$$ because both streams send gradients into it, and gradients are clipped to norm 10.

**Results** (57 games, percentage of human performance, 30 no-op starts): "Single" is Double DQN, and "Clip" adds gradient-norm clipping.

| Agent | Mean | Median |
|---|---|---|
| Nature DQN | 227.9% | 79.1% |
| Single (Double DQN) | 307.3% | 117.8% |
| Single Clip | 341.2% | 132.6% |
| Duel Clip | 373.1% | 151.5% |
| Prioritized Single | 434.6% | 123.7% |
| Prioritized Duel Clip | 591.9% | 172.1% |

Duel Clip beats Single Clip on 43 of 57 games (75.4%), and on 26 of the 30 games with 18 actions, consistent with the idea that dueling helps most when there are many similar actions.

### Multi-step learning: bootstrap from n steps ahead

**Source:** an old idea in RL (Sutton, 1988); Rainbow uses it as a DQN extension.

**The problem.** The 1-step target $$r + \gamma \max_{a'} Q(s', a')$$ uses one real reward and then relies on the network's own estimate. A reward found at the end of a long sequence propagates back only one step per backup, which is slow, and early on the bootstrapped part is mostly noise.

**The idea.** Accumulate n real rewards before bootstrapping:

$$
R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1}, \qquad
y_t = R_t^{(n)} + \gamma^n \max_{a'} Q(S_{t+n}, a'; \theta^-)
$$

and minimize $$\big(y_t - Q(S_t, A_t; \theta)\big)^2$$ as before. Rewards now propagate n steps per backup, and the target depends less on the current estimates. The price is variance (n random rewards instead of one) and, strictly, an off-policy bias: the n intermediate actions came from the older, exploratory behaviour policy, not the greedy one. In practice a small n works well; Rainbow compared n = 1, 3 and 5 and found **n = 3** best overall.

### Distributional RL (C51): learn the whole distribution of returns

**Paper:** Bellemare, Dabney and Munos (2017), *A Distributional Perspective on Reinforcement Learning*.

**The idea.** Q(s, a) is the **expected** return. Distributional RL learns the full random return Z(s, a), whose mean is Q, through a **distributional Bellman equation**:

$$
Z(s, a) \overset{D}{=} R(s, a) + \gamma Z(S', A')
$$

where $$\overset{D}{=}$$ means equal in distribution. Two returns can have the same mean but very different shapes: a "safe" action with return around 5, and a risky one that gives 10 or 0 with equal probability. The distribution keeps them apart.

**Representation.** The return distribution is modelled as a categorical distribution over **N = 51 fixed atoms** evenly spaced on [V_min, V_max] = [−10, 10]; hence the name **C51**. The network outputs, for each action, a softmax over the 51 atoms. Q(s, a) is recovered as the mean, $$\sum_i z_i\, p_i(s, a)$$, and actions are still chosen greedily on it.

**Training.** For a transition (s, a, r, s′):

1. Pick $$a^* = \arg\max_{a'} Q(s', a')$$ using the target network's distributions.
2. Move every atom by the Bellman update: $$\hat{\mathcal{T}} z_j = r + \gamma z_j$$, clipped to [V_min, V_max]. This shrinks and shifts the next-state distribution.
3. The shifted atoms no longer line up with the fixed support, so **project** each one's probability onto its two nearest neighbours on the grid, split in proportion to closeness (linear interpolation).
4. Minimize the **cross-entropy** between this projected target distribution and the predicted distribution $$p(s, a)$$, i.e. the KL divergence $$D_{\mathrm{KL}}(\Phi \hat{\mathcal{T}} Z_{\theta^-}(s,a) \,\Vert \, Z_\theta(s,a))$$.

The Bellman update becomes a multiclass classification problem. Everything else (target network, replay, ε-greedy on the mean) stays as in DQN.

**Theory.** For a fixed policy, the distributional Bellman operator is a γ-contraction in a maximal form of the Wasserstein metric, so the return distribution converges, not just its mean. For control (with the max), the operator is **not** a contraction in any metric between distributions, although the mean still converges to Q* as usual. In other words, the distributional view does not come with the same clean guarantees as Q-learning.

**Why it helps**, even though Atari is deterministic and only the mean is used to act: learning the full distribution is a much richer training signal (51 numbers per action instead of 1), it keeps separate modes from blurring together (for example "fire now and lose" versus "wait and survive"), and it appears to propagate rare events better; C51 did particularly well on sparse-reward games such as Venture and Private Eye.

**Results** (57 games, best evaluation score during training, 30 no-op starts; the mean for C51 is the corrected value from the paper's erratum):

| | DQN | Double DQN | Dueling | Prioritized | Prioritized Dueling | C51 |
|---|---|---|---|---|---|---|
| **Mean** | 228% | 307% | 373% | 434% | 592% | 701% |
| **Median** | 79% | 118% | 151% | 124% | 172% | 178% |
| **Games > human** | 24 | 33 | 37 | 39 | 39 | 40 |

C51 reached this using none of the other improvements, only DQN plus the distributional loss.

### Noisy Nets: learned, state-dependent exploration

**Paper:** Fortunato et al. (2018), *Noisy Networks for Exploration*.

**The problem.** ε-greedy explores by occasionally taking a uniformly random action, independently at each step. That is hopeless in games such as Montezuma's Revenge, where a long, specific sequence of actions is needed before the first reward. Random single-step dithering almost never produces such a sequence, and the amount of exploration is a hand-tuned schedule that is the same in every state.

**The idea.** Put the randomness in the **weights** instead of the actions. A standard linear layer y = wx + b is replaced by a **noisy linear layer**:

$$
y = (\mu^w + \sigma^w \odot \varepsilon^w)\, x + \mu^b + \sigma^b \odot \varepsilon^b
$$

where μ and σ are **learnable** parameters and ε is zero-mean noise resampled regularly. The network learns σ by gradient descent like any other weight: where exploring pays off, σ can stay large; where it does not, σ shrinks. Exploration becomes **state-dependent** and **self-annealing**, with no ε schedule to tune. Because the perturbation is applied to the weights, it stays consistent while a fixed noise sample is in use, which gives more coherent exploration than independent random actions.

**Details.** For DQN the fully connected layers are made noisy and ε-greedy is removed entirely: the agent acts greedily with respect to the noisy Q-network. To keep it cheap, the noise is **factorised**: rather than one Gaussian per weight (pq + q random numbers for p inputs and q outputs), it uses p + q unit Gaussians and sets $$\varepsilon^w_{ij} = f(\varepsilon_i) f(\varepsilon_j)$$ and $$\varepsilon^b_j = f(\varepsilon_j)$$ with $$f(x) = \operatorname{sgn}(x)\sqrt{\lvert x\rvert }$$. The σ parameters are initialized to $$\sigma_0/\sqrt{p}$$ with σ₀ = 0.5, and the noise is resampled before every action.

**Results** (57 games, human-normalized scores):

| | Baseline mean | Baseline median | NoisyNet mean | NoisyNet median | Median improvement |
|---|---|---|---|---|---|
| DQN | 319 | 83 | 379 | 123 | 48% |
| Dueling | 524 | 132 | 633 | 172 | 30% |
| A3C | 293 | 80 | 347 | 94 | 18% |

Interestingly, the average learned σ of the last noisy layer shrank over training in every game examined, but in the penultimate layer it **grew** in 3 of the 5 games: the agent does not necessarily converge toward a deterministic policy, and the exploration strategy it learns differs from game to game.

### Rainbow: combining everything

**Paper:** Hessel et al. (2018), *Rainbow: Combining Improvements in Deep Reinforcement Learning*.

The six extensions address different problems, so the natural question is whether they combine. Rainbow puts them all into one agent:

| Component | What it fixes | How it enters Rainbow |
|---|---|---|
| Double Q-learning | Overestimation from the max | Online network selects $$a^*_{t+n}$$, target network evaluates it |
| Prioritized replay | Uniform sampling wastes updates | Proportional, but priority is the **KL loss** instead of \|δ\| |
| Dueling networks | Every action's value learned separately | Value and advantage streams, each outputting 51 atoms per action |
| Multi-step learning | Slow reward propagation | n = 3-step returns |
| Distributional RL | Only the mean is learned | C51 with 51 atoms on [−10, 10] |
| Noisy Nets | Undirected ε-greedy exploration | All linear layers noisy (factorised), ε = 0 |

The pieces fit together in specific ways:

- **Multi-step distributional target:** the target distribution is the next-n-step distribution, shrunk by $$\gamma^n$$ and shifted by the n-step return, $$d_t^{(n)} = \big(R_t^{(n)} + \gamma^{n} z,\; p_{\theta^-}(S_{t+n}, a^*_{t+n})\big)$$, then projected onto the fixed support, and the loss is the KL divergence to the predicted distribution.
- **Priority = KL loss:** in the distributional setting the quantity being minimized is the KL loss, so it is used as the priority, $$p_t \propto \big(D_{\mathrm{KL}}\big)^{\omega}$$. It may be more robust in noisy environments, since the KL loss can keep decreasing even when returns are not deterministic.
- **Dueling over distributions:** for each atom, the value and advantage streams are combined dueling-style (subtracting the mean advantage) and then passed through a softmax over atoms to give each action's return distribution.

**Hyperparameters** (identical across all 57 games): Adam with learning rate 0.0000625 (a quarter of DQN's 0.00025) and Adam ϵ = 1.5 × 10⁻⁴; learning starts after 80K frames (prioritized replay allows starting earlier than DQN's 200K); target network period 32K frames; prioritization exponent ω = 0.5 with IS exponent β annealed from 0.4 to 1; n = 3; 51 atoms on [−10, 10]; Noisy Nets σ₀ = 0.5 and no ε-greedy.

**Results.** Median human-normalized scores of the best agent snapshot:

| Agent | No-op starts | Human starts |
|---|---|---|
| DQN | 79% | 68% |
| Double DQN | 117% | 110% |
| Prioritized Double DQN | 140% | 128% |
| Dueling Double DQN | 151% | 117% |
| A3C | – | 116% |
| Noisy DQN | 118% | 102% |
| Distributional DQN (C51) | 164% | 125% |
| **Rainbow** | **223%** | **153%** |

Rainbow matches DQN's **final** performance after only **7 million frames** (under 10 hours on one GPU), surpasses the best final performance of all the baselines within 44 million frames, and keeps improving up to 200 million.

**Ablations: which pieces matter most.** Removing one component at a time:

- **Prioritized replay and multi-step learning are the most crucial.** Removing either causes a large drop in median performance, early and (for multi-step) also at the end. The full agent beats both ablations on 53 of 57 games.
- **Distributional RL comes next.** For the first 40 million frames removing it makes no difference, but afterwards the ablated agent falls behind, mainly on games where performance is already near or above human level.
- **Noisy Nets** help in aggregate; removing them causes large drops on some games and small gains on others.
- **Dueling** and **Double Q-learning** make little difference to the median in this combination, helping on some games and hurting on others. For Double Q-learning the paper suggests a reason: with rewards clipped and the support limited to [−10, 10], true returns often exceed 10 and get clipped, so the agent tends to **underestimate** rather than overestimate, which removes the problem Double Q-learning was designed to fix. It may matter again with a wider support.

The broader lesson: the improvements are largely **complementary**, because they address different parts of the algorithm (the target, the sampling, the architecture, the output, the exploration). But they interact, and an idea that is essential on its own (Double DQN) can become nearly redundant in combination.

### Summary of the extensions

| Extension | Weakness of DQN it targets | Core change |
|---|---|---|
| Double DQN | Max overestimates values | Online net selects, target net evaluates |
| Prioritized replay | Uniform replay wastes updates | Sample ∝ \|TD error\|^α, correct with IS weights |
| Dueling | Value of each action learned separately | Q = V + (A − mean A) |
| Multi-step | Rewards propagate one step per backup | n-step returns before bootstrapping |
| Distributional (C51) | Only the expected return is learned | Categorical return distribution, KL loss |
| Noisy Nets | ε-greedy is undirected and state-independent | Learnable noise in the weights |
| Rainbow | — | All of the above together |

## 13. The whole algorithm in 60 seconds

Q-learning learns how good each action is by regressing toward a target: the reward plus the best value one step later. DQN does this with a convolutional network that looks at game pixels and outputs a score for every action.

Doing this naively is unstable for two reasons. First, the target is computed by the same network that is being trained, so every update moves the target too, like a dog chasing its own tail. DQN fixes this with a frozen copy of the network that produces the targets and is refreshed only every few thousand steps. Second, consecutive game frames are almost identical, so learning from them in order is like studying one page of a textbook a thousand times. DQN stores experiences in a large memory and trains on random samples from it.

Frozen targets plus shuffled memories turn an unstable problem into something close to ordinary supervised learning, and that is what let one algorithm learn dozens of Atari games from raw pixels.

## Sources

- Mnih, V. et al. (2015). [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). *Nature* 518, 529–533. The DQN algorithm, architecture, hyperparameters and ablations.
- Mnih, V. et al. (2013). [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602). arXiv:1312.5602. The earlier version with experience replay but without a separate target network.
- van Hasselt, H., Guez, A. and Silver, D. (2016). [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461). AAAI. Double DQN: Theorems 1 and 2, the value-estimate experiments, and Tables 1 and 2.
- van Hasselt, H. (2010). [Double Q-learning](https://papers.nips.cc/paper/3964-double-q-learning). NeurIPS. The double estimator.
- Thrun, S. and Schwartz, A. (1993). Issues in Using Function Approximation for Reinforcement Learning. *Proceedings of the Connectionist Models Summer School*. The first analysis of Q-learning's overestimation.
- Fujimoto, S., van Hoof, H. and Meger, D. (2018). [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477). ICML. TD3 and clipped double Q-learning.
- Schaul, T., Quan, J., Antonoglou, I. and Silver, D. (2016). [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952). ICLR. The Blind Cliffwalk, stochastic prioritization, importance sampling, Algorithm 1 and Table 1.
- Wang, Z. et al. (2016). [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581). ICML. Dueling networks and Table 1.
- Bellemare, M. G., Dabney, W. and Munos, R. (2017). [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887). ICML. C51, the projected distributional Bellman update and the contraction results.
- Fortunato, M. et al. (2018). [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295). ICLR. Noisy linear layers, factorised noise and Table 1.
- Hessel, M. et al. (2018). [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298). AAAI. The integrated agent, its hyperparameters, Table 2 and the ablations; also the switch from RMSProp to Adam.
- Sutton, R. S. (1988). [Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/article/10.1007/BF00115009). *Machine Learning* 3, 9–44. Multi-step (n-step) bootstrapping.
- Sutton, R. S. and Barto, A. G. (2018). [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), 2nd ed., Chapter 11. The deadly triad.
