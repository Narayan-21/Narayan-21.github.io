---
layout: post
title: "Maximum-Entropy RL: The 1-Step Problem"
date: 2026-10-01 10:00:00 +0530
description: Working through why an entropy bonus turns argmax into softmax and max into log-sum-exp, step by step, then following it into soft value iteration, SAC and robust RL.
tags: [RL]
toc:
  sidebar: left
related_posts: true
---

## Why I wrote this

I'm working through Pieter Abbeel's *Foundations of Deep RL* series, and Lecture 5 (DDPG and SAC) has a few slides on maximum-entropy RL that look short but hide a lot of algebra. This post is me filling in every step, mostly so that I actually understand it, and partly so the next person stuck on those slides has the full derivation in one place.

The one result I want to take away: adding an entropy bonus to a one-shot decision turns the greedy argmax policy into a softmax policy, and turns the value max r(a) into a smooth log-sum-exp.

$$
\pi^*(a) = \frac{1}{Z}\exp\!\Big(\frac{r(a)}{\beta}\Big), \qquad V^* = \beta \log \sum_a \exp\!\Big(\frac{r(a)}{\beta}\Big)
$$

I cover the two "Max-ent for 1-step problem" slides and the "Max-ent Value Iteration" slide, and then go into the paper by Eysenbach and Levine (2021) that explains why entropy makes policies robust. For me this was the stepping stone to soft Q-learning and Soft Actor-Critic.

**What I assume you know:** expectations over a discrete distribution, logs and exponentials, partial derivatives, and the basic idea of a Lagrange multiplier. No prior RL theory beyond "a policy is a distribution over actions" is needed.

**How the post is laid out:**

1. Sections 1 to 6 derive the optimal policy (slide 1).
2. Section 7 interprets the temperature β.
3. Sections 8 to 11 derive and interpret the optimal value V (slide 2).
4. Sections 12 and 13 give a second, shorter proof and a useful rewrite.
5. Section 14 derives max-ent value iteration (slide 3); section 15 extends it to discounting, soft Q-learning and SAC.
6. Sections 16 to 18 cover why entropy pays off: Eysenbach and Levine's proof that max-ent RL solves a robust RL problem, worked examples, and their experiments.
7. Section 19 puts everything on one page.

What helped me most was doing each derivation on paper before reading the next step, so if you're learning this too, I'd recommend the same.

## 1. The 1-step problem and why we add entropy

I like this setting because it strips RL down to one decision: no states, no future, one action a chosen from a discrete set, then a known reward r(a). It is a bandit with known rewards.

The policy π(a) is a probability distribution over actions, and the numbers π(a₁), π(a₂), … are what we optimize.

**Standard objective.** Maximize expected reward:

$$
\max_{\pi} \; E_{a \sim \pi}[r(a)]
$$

The answer is trivial: put probability 1 on the best action. The optimal policy is deterministic and greedy.

**Max-ent objective (top line of slide 1).** Add a bonus for the policy's entropy, weighted by a temperature β > 0:

$$
\max_{\pi(a)} \; E[r(a)] + \beta\,\mathcal{H}(\pi(a))
$$

Entropy measures how spread out the policy is. Rewarding it buys three things I care about in practice:

- **Exploration:** the policy keeps trying actions that currently look slightly worse.
- **Robustness:** it does not bet everything on one action whose reward estimate might be wrong.
- **Multiple good solutions:** when two actions are nearly equally good, it keeps both instead of picking one arbitrarily.

β sets the trade-off. Large β favors randomness; small β favors reward.

## 2. Writing the objective out explicitly

For a discrete distribution both terms become sums over actions.

$$
E[r(a)] = \sum_a \pi(a)\,r(a), \qquad \mathcal{H}(\pi) = -\sum_a \pi(a)\log\pi(a)
$$

The minus sign in the entropy is there because log π(a) ≤ 0, so each term −π log π is ≥ 0. Entropy is 0 for a deterministic policy and reaches its maximum, log \|A\|, for the uniform policy over \|A\| actions.

Substituting gives line 2 of slide 1:

$$
\max_{\pi(a)} \; \sum_a \pi(a)\,r(a) \;-\; \beta \sum_a \pi(a)\log\pi(a)
$$

The sign flip tripped me up the first time: "+β × entropy" becomes "−β × Σ π log π".

## 3. The constraint and the Lagrangian

The π(a) cannot be chosen freely; they must form a valid distribution.

$$
\sum_a \pi(a) = 1, \qquad \pi(a) \ge 0
$$

The equality constraint is handled with a Lagrange multiplier λ. This is line 3 of slide 1:

$$
\max_{\pi(a)} \min_{\lambda} \; \mathcal{L}(\pi, \lambda) = \sum_a \pi(a)r(a) - \beta\sum_a \pi(a)\log\pi(a) + \lambda\Big(\sum_a \pi(a) - 1\Big)
$$

**Why max over π, then min over λ?** This confused me at first. The trick is to look at the inner minimization for a fixed π:

- If π breaks the constraint, say Σ π > 1, the minimizer sends λ → −∞ and the Lagrangian goes to −∞. If Σ π &lt; 1, it sends λ → +∞ with the same effect. The maximizer will never pick such a π.
- If π satisfies the constraint, the bracket is 0, λ has no effect, and the Lagrangian equals the original objective.

So max over π of min over λ is exactly "maximize the objective over valid distributions". The constraint has been folded into an unconstrained problem.

**What about π(a) ≥ 0?** It never needs to be imposed. The derivative of −π log π goes to +∞ as π → 0, so the optimum is never on the boundary. The solution will also come out as an exponential, which is automatically positive.

## 4. Derivative with respect to π(a)

At the optimum, the partial derivative with respect to every variable is zero. Fix one specific action a and differentiate with respect to π(a).

The sums run over a dummy index; call it a′ to avoid confusion. Differentiating Σ over a′ with respect to π(a) keeps only the a′ = a term, because no other term depends on π(a).

**Reward term:**

$$
\frac{\partial}{\partial \pi(a)} \sum_{a'} \pi(a')\,r(a') = r(a)
$$

**Entropy term.** Use the product rule on π log π:

$$
\frac{d}{d\pi}\big[\pi\log\pi\big] = \log\pi + \pi\cdot\frac{1}{\pi} = \log\pi + 1
$$

so

$$
\frac{\partial}{\partial \pi(a)}\Big[-\beta\sum_{a'}\pi(a')\log\pi(a')\Big] = -\beta\log\pi(a) - \beta
$$

The "+1" from the product rule is where the stray −β on the slide comes from. I stared at that −β for a while before I saw it.

**Constraint term:**

$$
\frac{\partial}{\partial \pi(a)}\,\lambda\Big(\sum_{a'}\pi(a') - 1\Big) = \lambda
$$

**Setting the sum to zero** gives the stationarity condition on the slide:

$$
r(a) - \beta\log\pi(a) - \beta + \lambda = 0
$$

## 5. Solving for π(a) and finding Z

Rearrange the stationarity condition, divide by β and exponentiate:

$$
\beta\log\pi(a) = r(a) - \beta + \lambda \quad\Longrightarrow\quad \pi(a) = \exp\Big[\tfrac{1}{\beta}\big(r(a) - \beta + \lambda\big)\Big]
$$

Now split the exponent into the part that depends on a and the part that does not:

$$
\pi(a) = \underbrace{\exp\!\Big(\frac{r(a)}{\beta}\Big)}_{\text{depends on } a} \cdot \underbrace{\exp\!\Big(\frac{\lambda - \beta}{\beta}\Big)}_{\text{same constant } C \text{ for every } a}
$$

This was the key observation for me. λ − β is the same for every action, so it only rescales. The shape of the policy is fixed entirely by exp(r(a)/β).

**Using the λ condition.** Setting ∂L/∂λ = 0 (right column of slide 1) just returns the constraint Σ π(a) = 1. Plug in π(a) = C exp(r(a)/β):

$$
C\sum_a \exp\!\Big(\frac{r(a)}{\beta}\Big) = 1 \quad\Longrightarrow\quad C = \frac{1}{Z}, \qquad Z = \sum_a \exp\!\Big(\frac{r(a)}{\beta}\Big)
$$

**Result (bottom of slide 1):**

$$
\pi^*(a) = \frac{1}{Z}\exp\!\Big(\frac{r(a)}{\beta}\Big)
$$

Z is called the **partition function**, a name borrowed from statistical physics. If you want λ itself, C = exp((λ − β)/β) = 1/Z gives λ = β − β log Z. I never needed it again.

## 6. Checking it is the global maximum

A zero derivative only finds a stationary point; concavity makes it the unique global maximum.

- The reward term Σ π(a) r(a) is linear in π.
- Entropy is strictly concave. Its second derivative has no cross terms and is negative on the diagonal:

$$
\frac{\partial^2}{\partial \pi(a)^2}\big[-\pi(a)\log\pi(a)\big] = -\frac{1}{\pi(a)} < 0
$$

- With β > 0, linear plus β × strictly concave is strictly concave.
- The set of valid distributions (the probability simplex) is convex.

A strictly concave function on a convex set has at most one maximizer, and any stationary point is it. So π\* is the unique optimal policy.

## 7. Interpretation: the temperature β

The optimal policy is a softmax over rewards: higher reward gives exponentially higher probability, and β controls how sharp that preference is.

{% include notes/policy-vs-temperature.liquid caption="Computed from π(a) = exp(r(a)/β) / Z with r = [1, 2, 3], 11 values of β from 0.1 to 10" %}

**The two limits:**

- **β → 0:** the exponent r(a)/β blows up for the best action relative to the rest, so all probability goes to argmax r(a). This recovers ordinary greedy RL.
- **β → ∞:** every exponent goes to 0, every exp(·) goes to 1, and the policy becomes uniform. This is pure entropy maximization.

**Only reward differences matter.** Add a constant c to every r(a): the numerator and Z both get multiplied by exp(c/β), which cancels. This is why the probabilities depend on gaps like r(a₃) − r(a₂), not on absolute reward levels.

**Ratios between actions** make the role of β explicit:

$$
\frac{\pi(a_i)}{\pi(a_j)} = \exp\!\Big(\frac{r(a_i) - r(a_j)}{\beta}\Big)
$$

With β = 1, an action that is 1 reward unit better is e ≈ 2.72 times more likely. With β = 0.1 it is e¹⁰ ≈ 22,000 times more likely.

## 8. The optimal value V = β log Z

Plugging the optimal policy back into the objective gives V = β log Z (slide 2). V is the best achievable reward-plus-entropy; in ordinary RL the same quantity is just max r(a).

**Step 1: substitute π everywhere.** Every π(a) becomes (1/Z) exp(r(a)/β):

$$
V = \sum_a \tfrac{1}{Z}e^{r(a)/\beta}\,r(a) \;-\; \beta\sum_a \tfrac{1}{Z}e^{r(a)/\beta}\,\log\!\Big(\tfrac{1}{Z}e^{r(a)/\beta}\Big)
$$

**Step 2: split the log.** Use log(xy) = log x + log y:

$$
\log\!\Big(\tfrac{1}{Z}e^{r(a)/\beta}\Big) = \log\!\big(e^{r(a)/\beta}\big) + \log\tfrac{1}{Z}
$$

The first piece carries the same weight as the reward term, so the slide merges them into one sum. The second piece becomes its own sum:

$$
V = \sum_a \tfrac{1}{Z}e^{r(a)/\beta}\Big(r(a) - \beta\log e^{r(a)/\beta}\Big) \;-\; \beta\sum_a \tfrac{1}{Z}e^{r(a)/\beta}\log\tfrac{1}{Z}
$$

**Step 3: the first sum vanishes.** Inside the bracket, log and exp cancel:

$$
r(a) - \beta\log e^{r(a)/\beta} = r(a) - \beta\cdot\frac{r(a)}{\beta} = 0
$$

The reward exactly cancels part of the entropy, for every action and every β. In the second sum, log(1/Z) does not depend on a, so it moves outside:

$$
V = 0 - \beta\log\tfrac{1}{Z}\;\sum_a \tfrac{1}{Z}e^{r(a)/\beta}
$$

I misread the slide's line 3 at first. Read it as (β log 1/Z) × (Σ …), not as the log of the whole sum.

**Step 4: the remaining sum is 1.** It is Σ π(a), which equals 1 by the definition of Z:

$$
V = -\beta\log\tfrac{1}{Z}
$$

**Step 5: flip the sign and expand Z.** Since log(1/Z) = −log Z:

$$
V = \beta\log Z = \beta\log\sum_a \exp\!\Big(\frac{r(a)}{\beta}\Big)
$$

## 9. Soft max vs softmax, and the gradient

This one caught me out: the word "softmax" names two different objects, and the slide uses it for the value, not the policy.

| Object | Formula | What it smooths | Common names |
| --- | --- | --- | --- |
| Optimal value V | β log Σ exp(r(a)/β) | max r(a), a number | soft max, log-sum-exp (Abbeel's "softmax") |
| Optimal policy π | exp(r(a)/β) / Z | one-hot argmax, a distribution | softmax in ML libraries, soft-argmax, Boltzmann policy |

**The link: the gradient of log-sum-exp is the softmax distribution.** Differentiate V with respect to one reward:

$$
\frac{\partial V}{\partial r(a)} = \beta\cdot\frac{1}{Z}\cdot e^{r(a)/\beta}\cdot\frac{1}{\beta} = \pi(a)
$$

This mirrors the hard case, where the gradient of max r(a) is a one-hot vector on the argmax. Raising one action's reward by a small δ raises V by π(a) × δ: the value responds to each action in proportion to how often the optimal policy picks it.

## 10. How soft is the soft max? Bounds on V

V always lies between the best reward and the best reward plus β log \|A\|:

$$
r_{\max} \;\le\; V \;\le\; r_{\max} + \beta\log|A|
$$

**Lower bound.** Z contains the term exp(r\_max/β), and every other term is positive, so Z ≥ exp(r\_max/β). Taking β log of both sides gives V ≥ r\_max.

**Upper bound.** Each of the \|A\| terms is at most exp(r\_max/β), so Z ≤ \|A\| exp(r\_max/β). Taking β log gives V ≤ r\_max + β log \|A\|.

What this means:

- The max-ent agent never scores below greedy on its own objective, because it collects the entropy bonus on top.
- The extra is at most β log \|A\|, which is β times the largest possible entropy.
- As β → 0 the gap closes and V → max r(a). Max-ent RL contains ordinary RL as a limit.

## 11. Worked example

To make it concrete, I plugged in numbers. With rewards r = [1, 2, 3] and β = 1, the optimal policy is about [0.090, 0.245, 0.665] and V = log Z ≈ 3.408.

**Policy.** The exponentials are e¹ ≈ 2.718, e² ≈ 7.389 and e³ ≈ 20.086, so Z ≈ 30.19. Dividing each by Z gives π ≈ [0.090, 0.245, 0.665].

**Value from the formula.** V = 1 × log 30.19 ≈ 3.408.

**Value computed directly from the objective:**

| Quantity | Calculation | Result |
| --- | --- | --- |
| Expected reward | 0.090 × 1 + 0.245 × 2 + 0.665 × 3 | 2.575 |
| Entropy | −(0.090 ln 0.090 + 0.245 ln 0.245 + 0.665 ln 0.665) | 0.833 |
| Objective, β = 1 | 2.575 + 1 × 0.833 | 3.408 |

Both routes give 3.408, which was a nice sanity check.

**Bounds check.** 3 ≤ 3.408 ≤ 3 + ln 3 ≈ 4.099.

**The trade.** Compared with always picking action 3, the agent gives up 0.425 in expected reward and gains 0.833 in entropy. The net objective, 3.408, beats greedy's 3.0.

## 12. A second proof via KL divergence

After grinding through the Lagrangian, this shorter proof is the one that made it click for me. The whole result follows in one line, with no Lagrange multipliers, once you compare any policy π to the Boltzmann policy π\*.

Define π\*(a) = exp(r(a)/β) / Z, so log π\*(a) = r(a)/β − log Z. The KL divergence from π to π\* is:

$$
\mathrm{KL}(\pi \,\|\, \pi^*) = \sum_a \pi\log\pi - \sum_a \pi\log\pi^* = -\mathcal{H}(\pi) - \frac{1}{\beta}E_\pi[r] + \log Z
$$

Multiply by β and rearrange:

$$
E_\pi[r] + \beta\,\mathcal{H}(\pi) = \beta\log Z - \beta\,\mathrm{KL}(\pi \,\|\, \pi^*)
$$

The left side is exactly our objective. On the right, β log Z is a constant and KL ≥ 0, with equality only when π = π\*. So:

1. The objective is maximized exactly at π = π\* (sections 3 to 6).
2. The maximum value is β log Z (section 8).
3. Any other policy falls short by exactly β × KL(π ‖ π\*).

Point 3 is how SAC trains its policy network: it minimizes this KL to the Boltzmann distribution of its Q-values.

## 13. Writing the policy in terms of V

The log-probability of an action equals how far its reward sits above or below the soft value, divided by β.

Since V = β log Z, we have Z = exp(V/β). Substitute into the policy:

$$
\pi^*(a) = \frac{e^{r(a)/\beta}}{e^{V/\beta}} = \exp\!\Big(\frac{r(a) - V}{\beta}\Big) \quad\Longleftrightarrow\quad \beta\log\pi^*(a) = r(a) - V
$$

r(a) − V works like an advantage. Because V ≥ r\_max, every action has r(a) − V ≤ 0, which matches log π ≤ 0. This form is the one used directly in soft Q-learning.

## 14. Max-ent value iteration (slide 3)

The multi-step max-ent problem is solved by running the 1-step solution at every state, with Q in place of r, working backwards from the horizon. Abbeel's slide puts it as: the 1-step problem with Q instead of r, so the solution transcribes directly.

### The multi-step objective

Over a finite horizon H, the agent collects reward and an entropy bonus at every visited state:

$$
\max_{\pi} \; E\Big[\sum_{t=0}^{H} r_t + \beta\,\mathcal{H}\big(\pi(\cdot \mid s_t)\big)\Big]
$$

The expectation is over the trajectories the policy and the environment's dynamics produce. Entropy is not a one-time bonus: being random at step t is rewarded at step t.

### What V\_k(s) means

V\_k(s) is the best reward-plus-entropy you can collect starting in state s with k steps to go:

$$
V_k(s) = \max_{\pi} \; E\Big[\sum_{t=H-k}^{H} r(s_t, a_t) + \beta\,\mathcal{H}\big(\pi(\cdot \mid s_t)\big) \;\Big|\; s_{H-k} = s\Big]
$$

The sum starts at H − k because only the last steps remain. With nothing left to do, V₀(s) = 0 for every s.

### The recursion

The derivation splits the sum into the current step and everything after it.

1. The current step contributes r(s, a) + β H(π(·\|s)).
2. After acting, the agent lands in s′ with one step fewer to go. The best it can do from there is, by definition, V\_{k−1}(s′). The choice of policy now and the choices later are separate, so the later ones can be replaced by their optimal value.
3. Only the current-step policy π(·\|s) is left to optimize:

$$
V_k(s) = \max_{\pi(\cdot \mid s)} \; E_{a \sim \pi,\; s' \sim P(\cdot \mid s, a)}\Big[r(s,a) + \beta\,\mathcal{H}\big(\pi(\cdot \mid s)\big) + V_{k-1}(s')\Big]
$$

One thing I noticed: the slide's version closes the entropy bracket in the wrong place, after V\_{k−1}(s′). The entropy is only of π(·\|s); V\_{k−1}(s′) is added to it, as above.

{: start="4"}
4. The entropy term depends on the whole distribution π(·\|s), not on the sampled a or s′, so it moves outside the expectation. The other two terms are grouped and averaged over the next state:

$$
Q_k(s,a) = E_{s'}\big[r(s,a) + V_{k-1}(s')\big] = r(s,a) + \sum_{s'} P(s' \mid s,a)\,V_{k-1}(s')
$$

{: start="5"}
5. That leaves:

$$
V_k(s) = \max_{\pi(\cdot \mid s)} \; \sum_a \pi(a \mid s)\,Q_k(s,a) + \beta\,\mathcal{H}\big(\pi(\cdot \mid s)\big)
$$

### Why it is the 1-step problem

For a fixed state s and step k, the last line is the 1-step objective of section 1 with relabeled symbols:

| 1-step problem | Max-ent value iteration at state s, step k |
| --- | --- |
| Decision: π(a) | Decision: π(·\|s) |
| Payoff: r(a) | Payoff: Q\_k(s, a) |
| Optimal value: V | Optimal value: V\_k(s) |
| Normalizer: Z | Normalizer: Z\_k(s), one per state |

Everything derived in sections 3 to 8 carries over unchanged, so the solution is transcribed:

$$
V_k(s) = \beta\log\sum_a \exp\!\Big(\frac{Q_k(s,a)}{\beta}\Big), \qquad \pi_k(a \mid s) = \frac{1}{Z_k(s)}\exp\!\Big(\frac{Q_k(s,a)}{\beta}\Big) = \exp\!\Big(\frac{Q_k(s,a) - V_k(s)}{\beta}\Big)
$$

The rewrite on the right uses section 13. The bounds of section 10 also carry over: max over a of Q\_k(s, a) ≤ V\_k(s) ≤ max over a of Q\_k(s, a) + β log \|A\|.

### The algorithm

1. Set V₀(s) = 0 for every state.
2. For k = 1, 2, …, H, for every state s:

   1. Compute Q\_k(s, a) = r(s, a) + Σ over s′ of P(s′\|s, a) V\_{k−1}(s′) for every action.
   2. Compute V\_k(s) = β log Σ over a of exp(Q\_k(s, a)/β).
   3. Store π\_k(a\|s) = exp((Q\_k(s, a) − V\_k(s))/β).
3. To act, use π\_k where k is the number of steps remaining. The optimal finite-horizon policy changes as the deadline approaches.

Compared with ordinary value iteration, the only change is the max over actions becoming a log-sum-exp, and the argmax becoming a softmax. As β → 0 the algorithm becomes ordinary value iteration.

Like ordinary value iteration, this needs the transition model P to compute Q\_k. Section 15 shows how soft Q-learning replaces the model with sampled transitions.

### Worked example

One state with two actions that both loop back to it. Rewards are r = [0, 1], β = 1, and V₀ = 0.

| k | Q\_k(a₁) | Q\_k(a₂) | V\_k = log(e^Q₁ + e^Q₂) | π\_k(a₂) |
| --- | --- | --- | --- | --- |
| 1 | 0 + 0 = 0 | 1 + 0 = 1 | log(1 + e) ≈ 1.313 | e / (1 + e) ≈ 0.731 |
| 2 | 0 + 1.313 = 1.313 | 1 + 1.313 = 2.313 | 1.313 + log(1 + e) ≈ 2.627 | ≈ 0.731 |

V\_{k−1} adds the same amount to both actions here, so the policy stays the same (section 7: only reward differences matter), and V grows by about 1.313 per step. In an MDP with several states, the V\_{k−1}(s′) term differs by action, so actions that lead to better states gain probability.

## 15. Infinite horizon, soft Q-learning and SAC

In a full MDP, the 1-step result is applied at every state, with r(a) replaced by the soft Q-value Q(s, a).

$$
V(s) = \beta\log\sum_a \exp\!\Big(\frac{Q(s,a)}{\beta}\Big), \qquad \pi(a \mid s) = \exp\!\Big(\frac{Q(s,a) - V(s)}{\beta}\Big)
$$

$$
Q(s,a) = r(s,a) + \gamma\,E_{s'}\big[V(s')\big]
$$

Together these are the **soft Bellman equations**. Compare them with ordinary value iteration:

|  | Ordinary RL | Max-ent RL |
| --- | --- | --- |
| State value | V(s) = max over a of Q(s, a) | V(s) = β log Σ exp(Q(s, a)/β) |
| Optimal policy | argmax over a of Q(s, a) | softmax of Q(s, ·)/β |
| β → 0 limit | — | recovers the left column |

The hard max becomes the log-sum-exp derived in section 8; nothing else changes.

**How the algorithms use it:**

1. **Soft Q-learning** iterates the soft Bellman equations with a neural network for Q, computing V(s) by log-sum-exp over actions.
2. **Soft Actor-Critic (SAC)** handles continuous actions, where that sum cannot be computed. It trains a separate policy network to approximate exp((Q − V)/β) by minimizing the KL divergence from section 12.
3. **Automatic temperature tuning** in SAC adjusts β during training to hold the policy's entropy near a target.

## 16. Why entropy pays off: max-ent RL is robust RL

Section 1 listed robustness as a motivation, and I wanted to know whether that was more than a hand-wave. It is: Eysenbach and Levine (2021) prove that the Boltzmann policy derived above is also a robust policy. Its max-ent score on one reward is a guaranteed plain return on a whole set of perturbed rewards, and it lower-bounds the return under a set of perturbed dynamics.

The paper uses α for the temperature and sets α = 1 in its proofs. My β is the same thing.

### Robust RL in one line

Robust RL plays a two-player game: the agent picks a policy, then an adversary picks the worst environment from an allowed set.

$$
\max_{\pi} \; \min_{\tilde p \in \tilde{\mathcal P},\; \tilde r \in \tilde{\mathcal R}} \; E_{\pi, \tilde p}\Big[\sum_t \tilde r(s_t, a_t)\Big]
$$

No entropy appears here: the robust objective is plain return. What surprised me is that solving the max-ent problem solves this game for specific sets R̃ and P̃.

### The reward-robust set

The paper's Theorem 4.1 uses this set of rewards, which depends on the policy because the adversary may focus on states the policy actually visits:

$$
\tilde{\mathcal R}(\pi) = \Big\{ \tilde r \;:\; E_\pi\Big[\sum_t \log \int_{\mathcal A} \exp\big(r(s_t, a') - \tilde r(s_t, a')\big)\, da'\Big] \le \epsilon \Big\}
$$

The theorem states that the worst-case expected return over this set equals the max-ent objective J\_MaxEnt(π; p, r), for every policy with full support.

### Deriving it in the 1-step setting

The 1-step version needs only the tools from sections 3 and 9. Take discrete actions, β = 1, and write the adversary's reward as r̃(a) = r(a) − u(a). The set becomes a log-sum-exp budget on the perturbation u:

$$
\log\sum_a e^{u(a)} \le \epsilon
$$

The adversary wants to lower E\_π[r̃] = E\_π[r] − Σ π(a) u(a), so it maximizes Σ π(a) u(a) under that budget.

1. **Stationarity.** The gradient of Σ π u is π(a). The gradient of the log-sum-exp is the softmax of u (section 9). With multiplier μ, the optimum needs π(a) = μ · softmax(u)(a) for every a.
2. **Fix μ.** Summing over a gives 1 = μ, so softmax(u) = π, which means u(a) = log π(a) + c for some constant c.
3. **Use the budget.** The log-sum-exp of log π + c is c, so a tight budget gives c = ε.
4. **The adversary's damage.** Σ π u = Σ π log π + ε = ε − H(π).

So the worst-case return of any policy π is

$$
\min_{\tilde r \in \tilde{\mathcal R}} E_\pi[\tilde r] = E_\pi[r] + \mathcal H(\pi) - \epsilon, \qquad \tilde r^*(a) = r(a) - \log\pi(a) - \epsilon
$$

The right side is the max-ent objective minus a constant. At ε = 0 it is exactly the max-ent objective, which is the equality the paper states. A positive ε shifts every policy's score by the same ε, so it never changes which policy is best. Maximizing the robust objective is therefore maximizing the max-ent objective, and the answer is the Boltzmann policy from section 5.

The worst-case reward r − log π is the one the paper's reward-perturbation experiment attacks (section 18).

### How to read the guarantee

The reward you train on sits at the pessimistic edge of the set. At ε = 0, Σ exp(r − r̃) ≤ 1 forces r̃(a) > r(a) for every action, with less slack on some actions paid for by much more on others. The guarantee reads: whatever reward in the set turns out to be true, the policy's plain expected return is at least its max-ent score on r.

The paper is explicit about the flip side: max-ent RL is not robust with respect to the reward it is trained on. To get robustness for a task reward, you train on a deliberately pessimistic version of it (section 17).

This is the same Fenchel duality as section 12. Log-sum-exp and negative entropy are convex conjugates, which is why entropy appears whenever an adversary is limited by a log-sum-exp budget.

## 17. Worked robustness examples

On the paper's 2-armed bandit, the Boltzmann policy has the highest guaranteed return of any policy: 2.313, against 2.046 for a near-greedy policy.

### The 2-armed bandit (paper's Figure 2), with numbers

The paper uses rewards r(a₁) = 2 and r(a₂) = 1. Take β = 1 and ε = 0. Each policy's guarantee is E\_π[r] + H(π), and its worst-case reward is r − log π (section 16).

| Policy π | E\_π[r] | H(π) | Guaranteed return | Worst-case reward r̃\* |
| --- | --- | --- | --- | --- |
| Boltzmann [0.731, 0.269] | 1.731 | 0.582 | **2.313** | [2.313, 2.313] |
| [0.9, 0.1] | 1.900 | 0.325 | 2.225 | [2.105, 3.303] |
| Uniform [0.5, 0.5] | 1.500 | 0.693 | 2.193 | [2.693, 1.693] |
| Near-greedy [0.99, 0.01] | 1.990 | 0.056 | 2.046 | [2.010, 5.605] |

Three things stood out to me in this table:

- **The Boltzmann policy wins the robust game.** Its guarantee, 2.313, equals log(e² + e), its soft value V from section 8.
- **The greedy policy is fragile.** Its worst case is a reward where the arm it ignores is worth 5.6. Betting 99% on arm 1 leaves it exposed to exactly that.
- **Against the optimal policy, the adversary's best move is a flat reward.** r − log π\* = V for every action (section 13), so the worst case makes both arms worth 2.313. No reallocation hurts the policy any further.

### Dynamics robustness (paper's Theorem 4.2)

For perturbed dynamics, max-ent RL gives a lower bound rather than an equality. Train with a **pessimistic reward**, which needs r > 0:

$$
\bar r(s_t, a_t, s_{t+1}) = \frac{1}{T}\log r(s_t, a_t) + \mathcal H\big[s_{t+1} \mid s_t, a_t\big]
$$

The log compresses the reward, and the second term rewards states with noisy dynamics. When the dynamics noise is the same everywhere, that term is a constant and the reward is just log r.

The theorem then bounds the worst-case max-ent return under any dynamics p̃ in the robust set P̃(π):

$$
\min_{\tilde p \in \tilde{\mathcal P}(\pi)} J_{\text{MaxEnt}}(\pi; \tilde p, r) \;\ge\; \exp\big(J_{\text{MaxEnt}}(\pi; p, \bar r) + \log T\big)
$$

The proof turns a change of dynamics into a change of reward with importance weights and Jensen's inequality, then reuses the reward result. The adversary may choose a different p̃ at every step.

**How big is the robust set?** Lemma 4.3 says, for discrete actions:

$$
\epsilon = T \cdot E\Big[\mathcal H_{\tilde p}[s_{t+1} \mid s_t, a_t] + \mathcal H_\pi[a_t \mid s_t]\Big] \;\ge\; T \cdot E\big[\mathcal H_\pi[a_t \mid s_t]\big]
$$

The perturbation budget grows with the policy's action entropy. A more random policy is guaranteed against larger changes to the world. Since β controls the entropy (section 7), a higher temperature buys a larger set. The budget holds on average, so the adversary may spend it on a few large shocks or many small ones.

### Two examples from the paper

**Control penalty.** Actions lie in [−10, 10] and the reward is a task term minus (a − a\*)². The adversary moves the target action by Δa and halves the penalty weight. The paper's Gaussian-integral calculation places this inside the robust set when

$$
\Delta a^2 + \tfrac{1}{2}\log(2\pi) + \log 20 \le \epsilon / T
$$

so the policy tolerates target shifts on the order of √ε.

**Linear-Gaussian dynamics.** States and actions lie in [−10, 10], with p = N(As + Ba, 1) and reward ‖s‖². The adversary adds a bias b to the mean and widens the noise to √2. The pessimistic reward becomes (2/T) log ‖s\_t‖, and the perturbation is covered when

$$
\tfrac{1}{2} b^2 + \log(8\sqrt{\pi}) + \log 20 \le \epsilon
$$

The paper notes that if the adversary shifts the mean but keeps the variance at 1, the divergence is infinite and the guarantee says nothing. The set is unusual; section 18 lists this as a limitation.

### Temperature: one subtlety

The paper's appendix finds that, for reward perturbations, a higher temperature gives a smaller robust set: the policy is robust to fewer rewards, but ones closer to the training reward. For dynamics, Lemma 4.3 says the set grows with entropy, and the experiments find large temperatures help. The paper does not reconcile these two statements in one picture, so I read "higher β means more robust" as a dynamics-robustness claim.

## 18. What the paper's experiments show

Across robot manipulation, navigation and Gym control tasks, SAC (max-ent RL) held up under perturbations that broke standard RL, and matched purpose-built robust RL methods while being simpler. The standard-RL baseline is usually TD3; both are trained on the unperturbed task and tested on a changed one.

| Experiment (paper figure) | What changes at test time | Compared against | Result |
| --- | --- | --- | --- |
| Pusher (Fig. 1, 4a) | An obstacle is added in the path | TD3 | TD3 almost always hits the obstacle; SAC often goes around it |
| Button (Fig. 4b) | The box holding the button moves closer or farther; goal locations change | TD3 | SAC more robust to the moved box and goals |
| 2D navigation, L-shaped obstacle (Fig. 5) | None; tests whether max-ent is just noise | Standard RL + independent Gaussian action noise | Noise can avoid the obstacle only by entering costly red regions; SAC keeps entropy low early and raises it where it is safe |
| Puck push (Fig. 6) | The puck's XY position is knocked after 20 steps | SAC at several α | All succeed undisturbed; only the largest α stays robust to larger knocks |
| Peg insertion (Fig. 7) | Worst-case hole position, found by CMA-ES | TD3 | Both near 100% for small shifts; at 1.5 cm and 2 cm only SAC succeeds; beyond that neither does |
| Mass changes (Fig. 3) | Body masses differ from training | PR-MDP, NR-MDP (Tessler et al., 2019), DDPG | SAC with large α is "at least competitive, if not better"; 30 evaluation episodes × 5 seeds |
| Reward perturbation (Fig. 8, 11) | Worst-case reward r̃ = r − log π | SVG-0, fictitious play, on 4 Gym tasks | All maximize ordinary return; only SAC also maximizes the worst-case reward |

The exact success rates and returns are plotted in the figures; the text quoted above gives the qualitative outcome and the thresholds it states.

**For me, the L-shaped navigation result is the key conceptual one.** Max-ent RL is not standard RL plus noise. Entropy enters the objective (section 1), so the policy learns where randomness is cheap and where it is costly, and becomes random only where it can afford to be. Uniform added noise cannot make that distinction.

### Limitations the authors state

- The robust sets are not the familiar H∞ or KL-ball sets, so there is no claim of robustness to those.
- The pessimistic reward for dynamics needs the dynamics entropy, which is hard to estimate in general MDPs.
- The temperature α still has to be chosen, and robustness to large perturbations needs a fairly large one.
- Max-ent RL is presented as a simple, decent robust method, not the best one.

## 19. Summary

Here is everything from this post on one page.

| Item | Result |
| --- | --- |
| Objective | max over π of Σ π r − β Σ π log π, with Σ π = 1 |
| Stationarity | r(a) − β log π(a) − β + λ = 0 |
| Optimal policy | π\*(a) = exp(r(a)/β) / Z |
| Partition function | Z = Σ exp(r(a)/β) |
| Optimal value | V = β log Z (log-sum-exp, the "soft max") |
| Policy via value | β log π\*(a) = r(a) − V |
| Gradient | ∂V/∂r(a) = π\*(a) |
| Bounds | r\_max ≤ V ≤ r\_max + β log \|A\| |
| Gap of any policy | objective(π) = V − β KL(π ‖ π\*) |
| Limits | β → 0: greedy, V → max r · β → ∞: uniform |

**Robustness, sections 16 to 18:**

| Item | Result |
| --- | --- |
| Robust RL | max over π, min over perturbed r̃, p̃ of plain expected return |
| Reward robust set (1-step, β = 1) | log Σ exp(r − r̃) ≤ ε |
| Worst-case return | E\_π[r] + H(π) − ε, so the Boltzmann policy is the robust policy |
| Worst-case reward | r̃\* = r − log π − ε; flat at V against the optimal policy |
| Dynamics | max-ent on the pessimistic reward (1/T) log r + H[s′\|s, a] lower-bounds the worst case |
| Set size | ε ≥ T × average action entropy: more entropy, larger perturbations covered |
| Price | lower nominal performance; α must be tuned |

## Wrapping up

What stuck with me most is how little actually changes. Adding an entropy term turns every hard max into a log-sum-exp and every argmax into a softmax, and the rest of RL carries over. The same small change is what makes SAC work, and, as Eysenbach and Levine show, it also buys a precise kind of robustness for free.

If you spot a mistake in any of the derivations, I'd love to hear about it.

## Sources

- Eysenbach, B. and Levine, S. (2021). [Maximum Entropy RL (Provably) Solves Some Robust RL Problems](https://arxiv.org/abs/2103.06257). arXiv:2103.06257. Sections 16 to 18.
- Pieter Abbeel, *Foundations of Deep RL*, Lecture 5 slides "Max-ent for 1-step problem" and "Max-ent Value Iteration". Sections 1 to 14.
