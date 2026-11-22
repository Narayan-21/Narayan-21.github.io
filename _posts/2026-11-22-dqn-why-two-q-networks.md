---
layout: post
title: "Why DQN Uses Two Q-Networks"
date: 2026-11-22 10:30:00 +0530
description: The moving-target problem in deep Q-learning, worked through with numbers - why DQN keeps a frozen target network, what it buys, what can still go wrong while it is frozen, and how Double DQN reuses it.
tags: [RL]
toc:
  sidebar: left
related_posts: true
---

## Overview

The Deep Q-Network (DQN) algorithm keeps **two copies** of its Q-network:

- the **online network** $$Q(\cdot\,; \theta)$$, which is trained at every step and used to choose actions, and
- the **target network** $$\hat{Q}(\cdot\,; \theta^-)$$, which is never trained by gradient descent. It only computes the regression targets, and every C steps (C = 10,000 in the Nature paper) it is overwritten with a copy of the online weights.

That looks odd at first. Q-learning has a single update rule and a single Q-function, so why two networks? This post answers that question in depth, with a small numerical example that shows exactly what goes wrong with one network and what the frozen copy fixes. The full DQN algorithm, including experience replay and the Atari details, is covered in [Deep Q-Networks: How Deep Q-Learning Learned to Play Atari]({% post_url 2026-11-22-dqn-deep-q-learning-atari %}).

**The short answer:** Q-learning has to write its own training labels from its current guesses. With a neural network, every update to the prediction also changes those labels, so the network chases a target that moves every time it steps toward it. DQN computes the labels from an old, frozen copy of the network so that they stay still long enough to be learned.

## 1. The two Q's were already there

Start from approximate Q-learning with a parameterized Q-function $$Q_\theta$$. After observing a transition (s, a, r, s′), compute a target with the current parameters θ_k, then take a gradient step on the squared error:

$$
\text{target}(s') = r + \gamma \max_{a'} Q_{\theta_k}(s', a')
$$

$$
\theta_{k+1} \leftarrow \theta_k - \alpha \,\nabla_\theta \Big[\tfrac{1}{2}\big(Q_\theta(s,a) - \text{target}(s')\big)^2\Big]_{\theta=\theta_k}
$$

The gradient is taken with respect to θ at θ = θ_k, and only through the prediction. Two different roles are hiding in this one rule:

| Role | Approximate Q-learning | DQN |
|---|---|---|
| **Prediction** (gets gradients) | $$Q_\theta(s,a)$$ | $$Q(\phi_j, a_j; \theta)$$, the online network |
| **Target** (treated as a fixed label) | $$Q_{\theta_k}(s', a')$$ | $$\hat{Q}(\phi_{j+1}, a'; \theta^-)$$, the target network |

Here φ_j is the preprocessed state (a stack of the last four 84×84 frames in Atari), and the DQN target is

$$
y_j = \begin{cases} r_j & \text{if } s_{j+1} \text{ is terminal} \\ r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^-) & \text{otherwise} \end{cases}
$$

In plain approximate Q-learning, the "label" parameters are refreshed after every single update, so they always equal the current θ. DQN lets them **lag behind**: θ⁻ is frozen and copied from θ only every C steps.

So DQN did not invent a new update rule. It is the same squared-error regression toward a Bellman target. **The only change is how stale the target parameters are allowed to be.**

### A different kind of "multiple Q-values"

There is a second, unrelated sense in which DQN has "many Q-values". The network takes a state and outputs **one Q-value per action**, up to 18 numbers for Atari from a single forward pass. That is just an efficient architecture: it makes $$\arg\max_a Q(\phi, a; \theta)$$ and $$\max_{a'} \hat{Q}(\phi', a'; \theta^-)$$ cost one forward pass each. During training only the output for the action actually taken, $$a_j$$, gets an error signal. This is not the reason for the two networks, and the rest of this post is about θ versus θ⁻.

## 2. Supervised learning has an answer key. Q-learning writes its own.

In normal supervised learning, say a cat-vs-dog classifier, every image comes with a label from the dataset. The label never changes. The network keeps moving its prediction toward that fixed answer, and gradient descent is good at this.

Q-learning has **no dataset of correct answers**. Nobody says "the true value of pressing LEFT here is 7.3". So the network writes its own label:

$$
\text{label} = r + \gamma \cdot \max_{a'} Q(s', a')
$$

In words: *the reward just received, plus what the network currently believes the next state is worth.* The second part comes from the network itself. **The network both writes the answer key and studies from it.**

That alone is not fatal. In a table it works fine, because updates are **local**: changing the entry Q(s, a) leaves Q(s′, a′) alone, so the label stays put while the prediction moves toward it.

In a neural network, updates are **global**. Changing θ to raise Q(s, a) also changes the outputs for every similar input. In Atari, s and s′ are consecutive frames that are nearly pixel-identical; maybe the ball moved two pixels. To a neural network they are about as similar as two inputs can be. So when the update pushes Q(s, a) up, Q(s′, ·) goes up too, the label goes up, the next update pushes Q(s, a) up again, and so on.

A useful picture: shooting arrows at a target that is tied to the bow. Every time you aim, the target moves with you.

## 3. Worked example: one network chasing its own tail

Take a toy case with γ = 0.9. The agent is in state s, takes action a, gets reward r = 1, and lands in state s′. The network currently says Q(s, a) = 5 and $$\max_{a'} Q(s', a') = 5$$ (written Q(s′) below).

**Step 1.** Label = 1 + 0.9 × 5 = **5.5**. The prediction is 5, so the update adjusts the weights to push Q(s, a) toward 5.5.

Because s and s′ look almost identical, the same weight change raises Q(s′) as well. Suppose, for simplicity, it moves by the same amount, to 5.5.

**Step 2.** Label = 1 + 0.9 × 5.5 = **5.95**. The answer key changed *because the network studied*.

**Step 3.** The update pushes toward 5.95, Q(s′) rises again, and the label becomes 1 + 0.9 × 5.95 ≈ **6.36**. And so on.

| Step | Prediction Q(s, a) | Q(s′) (same network) | Label |
|---|---|---|---|
| 1 | 5.0 | 5.0 | 5.5 |
| 2 | 5.5 | 5.5 ← dragged up | 5.95 |
| 3 | 5.95 | 5.95 ← dragged up | 6.36 |

Every update moves the thing being aimed at.

In this tiny example the values still drift somewhere sensible (here toward 1 / (1 − 0.9) = 10). In a real network, though, one gradient step on a minibatch changes the labels for thousands of states at once, by amounts that are hard to predict and not tied to how much the prediction moved. A small error immediately becomes part of the labels, gets learned, makes the next labels worse, and so on. That feedback loop, label changes driven by the learner's own last step, is why Q-learning with neural networks used to oscillate or blow up to huge values.

## 4. The fix: study from a photocopy of the answer key

DQN's fix is to **make a frozen copy of the network and use only the copy to write labels.**

- **Online network, Q with weights θ:** the student. It is updated every step and is always the latest version. It is used to choose actions and to make predictions.
- **Target network, $$\hat{Q}$$ with weights θ⁻:** a photocopy of the student from a while ago. It is **not trained**. Its only job is to compute labels.

The same toy example, now with a frozen copy:

| Step | Online Q(s, a) | Online Q(s′) | Frozen copy's Q(s′) | Label |
|---|---|---|---|---|
| 1 | 5.0 | 5.0 | 5.0 | 5.5 |
| 2 | 5.5 | 5.5 (moved) | **5.0** (frozen) | **5.5** |
| 3 | 5.5 | 5.5 | **5.0** | **5.5** |

The online network's Q(s′) still changes, but the label does not care, because it is computed from the frozen copy. **For a while, the answer key stands still**, and the problem behaves like normal supervised learning with fixed labels.

**Every C steps** the online weights are copied into the frozen copy. Now the copy says Q(s′) = 5.5, the new label is 1 + 0.9 × 5.5 = 5.95, and the online network studies toward that for the next C steps. Then the copy is refreshed again:

| Copy number | Frozen Q(s′) | Label for the next C steps |
|---|---|---|
| 0 | 5.0 | 5.5 |
| 1 | 5.5 | 5.95 |
| 2 | 5.95 | 6.36 |
| 3 | 6.36 | 6.72 |
| … | … | … |
| ∞ | 10 | 10 |

The labels are the same numbers as in the single-network table, but they now change **only at copy time**, and each one is held fixed long enough to be learned properly. The values improve in stages rather than in a continuous self-reinforcing loop. Each stage is a contraction toward the fixed point Q = 1 + 0.9 Q, i.e. Q = 10, and the gap shrinks by a factor of γ = 0.9 per copy.

### Which one lags, and why that is good

The **target network θ⁻ lags**. It is always an old snapshot of the online network, up to C steps old. The online network θ is always current.

The lag is the whole point. A mistake the online network makes right now **cannot affect the labels until the next copy**. By then, thousands of other updates have had a chance to average that mistake out. Without the lag, the mistake would feed straight into the very next label.

### The cost: a knob between stable and slow

Because labels come from an older network, new knowledge spreads more slowly. A reward discovered at the end of a long corridor needs one copy per step of the corridor to propagate back to the start.

- If C is very large, learning is stable but slow.
- If C is very small (C = 1 means no target network at all), the agent is back to chasing its own tail.

C = 10,000 was the value that worked across the Atari games.

## 5. The deeper view: a sequence of supervised problems

Between two copies, the labels $$y = r + \gamma \max_{a'} \hat{Q}(s', a'; \theta^-)$$ are a fixed function of the data. Gradient descent on the online network is therefore solving

$$
\theta \approx \arg\min_\theta \; \mathbb{E}_{(s,a,r,s') \sim D}\Big[\big(r + \gamma \max_{a'} \hat{Q}(s', a'; \theta^-) - Q(s, a; \theta)\big)^2\Big]
$$

whose minimizer (if the network could represent it exactly) is the Bellman optimality backup of the frozen network:

$$
Q_{\text{new}}(s,a) \approx \mathbb{E}\big[\, r + \gamma \max_{a'} Q_{\text{old}}(s', a') \,\big|\, s, a\big] = (\mathcal{T} Q_{\text{old}})(s,a)
$$

Then the copy sets $$Q_{\text{old}} \leftarrow Q_{\text{new}}$$ and the next window starts. So each sync is roughly **one step of value iteration**, $$Q_{k+1} \approx \mathcal{T} Q_k$$, and DQN is approximately a sequence of supervised regression problems, each bootstrapped from the previous answer. This is the same idea as **fitted Q-iteration**. Value iteration converges because $$\mathcal{T}$$ is a γ-contraction; the frozen target is what makes DQN look like value iteration rather than like a learner chasing its own updates.

## 6. While the copy is frozen, can the online network still run away?

A natural follow-up: the copy is refreshed from the online network every 10,000 steps, so what stops the online network itself from going out of control *during* those 10,000 steps?

**Mostly, nothing can run away**, because the mechanism that made it run away is switched off. But the loop is not gone completely, it is just much slower. Both halves of that answer matter.

### Why the online network stays in control between copies

What made things explode before was a loop:

**online changes → labels change → online changes more → labels change more.**

During the frozen window the labels come only from the frozen copy:

$$
y = r + \gamma \max_{a'} \hat{Q}(s', a'; \theta^-)
$$

Whatever the online network does, these numbers do not change, so there is nothing for it to chase. Its job for those 10,000 steps is ordinary supervised learning: fixed (input, label) pairs, fit them. Supervised learning on fixed labels is something gradient descent handles well with a sensible learning rate.

In the toy example: if the frozen copy says the label is 5.5, the online network moves toward 5.5 and **stops there**. Overshooting to 6 gives a gradient that pushes it back down to 5.5. **A fixed target pulls the prediction back; a moving target runs away.** That is the key difference.

Several other design choices keep each individual step small during the window:

- **Reward clipping to [−1, +1]** keeps the labels themselves small.
- **Huber loss (error clipping)** means a large error gives a capped gradient, so one unusual transition cannot cause a huge jump.
- **A small learning rate** (0.00025 in the paper) and **minibatches of 32 random samples** mean each step is a small, averaged nudge.

### What can still go wrong

1. **Errors can still pass from one copy to the next.** If the frozen copy has a mistake, say it overvalues some state, the online network faithfully learns that mistake, and at the next copy the mistake becomes part of the new labels. So the old loop still exists, but it turns once every 10,000 steps instead of every step. That gives many other updates time to average the error out, so it usually shrinks instead of growing. Usually, not always.

2. **The max overestimates.** The label takes a max over noisy Q-values, so it tends to pick whichever action is accidentally overestimated. That optimism leaks in at every copy, and Q-values can creep upward over many copies. Double DQN (section 7) was designed to fix exactly this.

3. **The online network also picks actions.** If its values go wrong, it plays differently, and that changes what data is collected. The replay buffer holds a million past transitions, so new data only slowly changes what is being trained on, but this loop is not fully closed either.

4. **Plain optimization problems.** If the learning rate is too high, even regression on fixed labels can be unstable, as in any neural network.

5. **Generalization can make the fitted backup itself expand.** The value-iteration view in section 5 assumes the network can fit the backup exactly. With function approximation, fitting one state's label can move other states' values by *more* than it moves the fitted one. A classic two-state example from Sutton and Barto (Section 11.2) makes this concrete. Suppose a linear model gives state s the value w and state s′ the value 2w (the two states share one weight, with features 1 and 2), the only transition trained on is s → s′ with reward 0, and γ = 0.9. With a frozen copy, each window fits

   $$
   w_{\text{new}} = 0 + 0.9 \times 2\, w_{\text{old}} = 1.8\, w_{\text{old}}
   $$

   Starting from w = 1, the copies give 1, 1.8, 3.24, 5.83, 10.5, … and the value diverges even though every window is a perfectly stable regression. The target network removed the fast, step-by-step loop, but it cannot fix a backup that, through generalization, is no longer a contraction. This is the **deadly triad** (function approximation + bootstrapping + off-policy data) in its smallest form. In Sutton and Barto's version, the divergence goes away if transitions *out of* s′ are also trained on as often as an on-policy agent would visit them. That is part of why the data distribution, and therefore what goes into the replay buffer, matters too.

So DQN does not **guarantee** stability. It is a set of tricks that make instability rare in practice, and occasional divergence was still observed. Later methods add more safeguards:

- **Double DQN** reduces the overestimation from the max.
- **Gradient-norm clipping** caps the size of any single update.
- **Soft (Polyak) target updates.** Instead of copying everything at once every C steps, the target network slowly blends toward the online network at every step:

  $$
  \theta^- \leftarrow (1 - \tau)\,\theta^- + \tau\,\theta, \qquad \text{e.g. } \tau = 0.005
  $$

  The labels drift smoothly instead of jumping. DDPG and SAC use this. With τ = 0.005 the target network's effective memory is about 1/τ = 200 steps, which plays the same role as C.

## 7. A bonus use for the second network: Double DQN

Once there are two networks, there is a second job they can do together.

The max in the target, $$\max_{a'} \hat{Q}(s', a'; \theta^-)$$, both **chooses** the best next action and **evaluates** it with the same noisy estimates. If the estimates are noisy, the chosen action is likely one whose value happens to be overestimated, so

$$
\mathbb{E}\Big[\max_{a'} \hat{Q}(s', a')\Big] \;\ge\; \max_{a'} \mathbb{E}\Big[\hat{Q}(s', a')\Big]
$$

and the bias is upward.

**A small numerical illustration.** Suppose s′ has three actions whose true values are all 0, and each estimate has independent noise that is equally likely to be +1 or −1. The max of three such estimates is +1 unless all three are −1, which happens with probability 1/8. So the expected max is 7/8 × 1 + 1/8 × (−1) = **0.75**, although the true max is **0**. Every label built from such a max is too optimistic, and bootstrapping passes the optimism on.

**Double DQN** (van Hasselt et al., 2016) separates the two jobs across the two networks DQN already has. The online network *chooses* the action and the target network *evaluates* it:

$$
y = r + \gamma\, \hat{Q}\big(s', \arg\max_{a'} Q(s', a'; \theta);\; \theta^-\big)
$$

If the online network picks an action because its own noise made it look good, the target network's estimate of that same action has partly independent noise, so on average it is not inflated. In the three-action example with fully independent noise, the evaluation of the chosen action is +1 or −1 with equal probability, so its expected value is **0**, the true value. In practice the two networks' errors are only partly independent, so the bias is reduced rather than removed, but it is a one-line change that uses a network DQN already had for free.

## 8. Summary

| Question | Answer |
|---|---|
| What are the two networks? | Online Q(·; θ), trained every step and used for acting; target $$\hat{Q}$$(·; θ⁻), never trained, only computes labels. |
| Is this a new update rule? | No. It is approximate Q-learning with the target parameters allowed to lag for C steps instead of being refreshed after every update. |
| What problem does it solve? | With one network, updating Q(s, a) also moves Q(s′, ·) through generalization, so the label moves every time the network steps toward it. |
| Why does freezing help? | Between copies the labels are fixed, so learning is ordinary supervised regression; errors reach the labels only at copy time, after many updates have averaged them out. |
| What is the cost? | New information propagates more slowly. C trades stability against speed. |
| What is the deeper view? | Each copy is roughly one step of value iteration; DQN approximates fitted Q-iteration. |
| Is stability guaranteed? | No. Errors can still carry across copies, the max overestimates, the policy shapes the data, and generalization can make the backup expand (the deadly triad). |
| What else do the two networks enable? | Double DQN: the online network chooses the next action, the target network evaluates it, which reduces overestimation. |

**In one sentence:** the network has to make up its own training labels from its current guesses, so DQN keeps an old frozen copy to write those labels; otherwise every update would shift the target it is aiming at, and every few thousand steps the copy is refreshed with what the network has learned.

## Sources

- Mnih, V. et al. (2015). [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). *Nature* 518, 529–533. The target network, C = 10,000 and the other hyperparameters.
- van Hasselt, H., Guez, A. and Silver, D. (2016). [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461). AAAI. Double DQN and the overestimation analysis.
- Lillicrap, T. et al. (2016). [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971). ICLR. Soft target updates in DDPG.
- Riedmiller, M. (2005). [Neural Fitted Q Iteration](https://link.springer.com/chapter/10.1007/11564096_32). ECML. Fitted Q-iteration with neural networks.
- Sutton, R. S. and Barto, A. G. (2018). [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), 2nd ed., Section 11.2 and Chapter 11. The w → 2w divergence example and the deadly triad.
