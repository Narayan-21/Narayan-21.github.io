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
    <text x="621" y="55">|A| outputs</text>
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
L_\delta(a) = \begin{cases} \tfrac{1}{2} a^2 & \text{for } |a| \le \delta, \\[4pt] \delta\big(|a| - \tfrac{1}{2}\delta\big) & \text{otherwise.} \end{cases}
$$

It is quadratic near zero and linear in the tails, and the two pieces meet with the same value and the same slope at |a| = δ. Its derivative is what matters for training:

$$
\frac{\partial L_\delta}{\partial a} = \begin{cases} a & \text{for } |a| \le \delta, \\ \delta \cdot \operatorname{sign}(a) & \text{otherwise,} \end{cases}
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
<figcaption style="text-align: center; font-size: 0.9em;">Squared loss versus Huber loss (δ = 1). They are identical for |a| ≤ 1; beyond that the Huber loss grows only linearly.</figcaption>
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

The max in the target is **biased upward**. If the Q-estimates are noisy, $$\max_{a'}$$ picks whichever action happens to be overestimated, so $$\mathbb{E}[\max_{a'} \hat{Q}] \ge \max_{a'} \mathbb{E}[\hat{Q}]$$. That optimism then compounds through bootstrapping, because overestimated values become part of the next labels.

**Double DQN** (van Hasselt et al., 2016) fixes this with the two networks DQN already has. The online network *chooses* the next action and the target network *evaluates* it:

$$
y = r + \gamma\, \hat{Q}\big(s', \arg\max_{a'} Q(s', a'; \theta);\; \theta^-\big)
$$

Because the two networks' errors are partly independent, an action that the online network overestimates by chance is usually not also overestimated by the target network, and the bias largely cancels. It is a one-line change to the target computation.

## 11. The whole algorithm in 60 seconds

Q-learning learns how good each action is by regressing toward a target: the reward plus the best value one step later. DQN does this with a convolutional network that looks at game pixels and outputs a score for every action.

Doing this naively is unstable for two reasons. First, the target is computed by the same network that is being trained, so every update moves the target too, like a dog chasing its own tail. DQN fixes this with a frozen copy of the network that produces the targets and is refreshed only every few thousand steps. Second, consecutive game frames are almost identical, so learning from them in order is like studying one page of a textbook a thousand times. DQN stores experiences in a large memory and trains on random samples from it.

Frozen targets plus shuffled memories turn an unstable problem into something close to ordinary supervised learning, and that is what let one algorithm learn dozens of Atari games from raw pixels.

## Sources

- Mnih, V. et al. (2015). [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). *Nature* 518, 529–533. The DQN algorithm, architecture, hyperparameters and ablations.
- Mnih, V. et al. (2013). [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602). arXiv:1312.5602. The earlier version with experience replay but without a separate target network.
- van Hasselt, H., Guez, A. and Silver, D. (2016). [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461). AAAI. Double DQN.
- Hessel, M. et al. (2018). [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298). AAAI. Adam instead of RMSProp, among other improvements.
- Sutton, R. S. and Barto, A. G. (2018). [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), 2nd ed., Chapter 11. The deadly triad.
