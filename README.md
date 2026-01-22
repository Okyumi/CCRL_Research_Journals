# CCRL_Research_Journals
## Research Log 2026

This repository mainly serves as a tracking log for my ongoing research project, for future reference.




### General Takeaways So Far

> [!NOTE]
>  
> **Deep understanding matters more than anything else.**  
>  
> One of the most important things I learned from my past research experience is the value of deep understanding in research, which is, never pretend that you understand. When you study a problem, you should not stop at the level of repeating someone else’s explanation. You should understand every part of it. If anything feels confusing, you should not pretend you understand. You should keep asking questions, again and again, unless you can barely find any questions. For the topic you are working on, you need to know it more deeply than anyone else.  
>  
> In practice, this means you should be able to answer questions clearly, no matter who asks them. If you truly understand something, you can explain it in the simplest language, and ideally you can walk others through a few concrete examples. The reason you can answer questions well is that, before others even ask, you should already anticipate many of the questions during your own thinking and research. Then you can use papers, references, AI tools, and your own reasoning to work toward solid answers. You also need to stay alert to any hidden assumptions behind an answer.  
>  
> The most important mindset is to be absolutely honest about what you know and what you do not know. Do not settle for shallow understanding. Face your gaps and ignorance directly, and do not be embarrassed to ask. Keep digging until you get to the bottom of the issue, and build a truly deep grasp of your research. This is also how he guided us. For every statement we made and every algorithm we proposed, he would try to understand it very thoroughly, not in a vague or superficial way, and never with empty, formal sounding talk (that I had with some other professor I had worked with).  
>
> **First-principles thinking is essential.**  
>  
> The second thing I learned what is often called first principles thinking. It means setting aside what others have done and not getting trapped by the current status quo. Instead, you go back to the original starting point and focus on the real problem you are trying to solve. From there, you try to solve the problem itself, rather than simply building on top of someone else’s solution.  
>
> **Think top-down, execute bottom-up.**  
>  
> From that foundation, we often use a top down approach. First, we state the problem clearly. That becomes the vision. Then we ask what directions might lead to a solution. Under each direction, we keep decomposing until the work becomes a set of small, executable tasks. Then we start with the next executable step.
>
>
> **Productivity follows physics.**  
>  
> Newton’s First Law of Motion states that an object will remain at rest or move at a constant velocity unless acted upon by an external force. The same applies to productivity. Often, the hardest part of a task is simply starting, because kinetic friction is less than static friction.  
>  
> Your next executable step should be small, because smaller steps reduce friction and make it easier to start. For example, if I want to run experiments, I may first need to modify code. Then my next executable step could be as simple as opening my laptop lid. After that, opening VS Code or PyCharm or Cursor, connecting to the HPC or the cloud, and continuing from there. Each step should be specific, small, and something you can hold yourself accountable to.  


---
Before introducing the continual learning pipeline, we define the core objective functions based on the probabilistic view of reinforcement learning.

### Goal-Conditioned Reward

We define the reward function as the probability density of reaching a goal $g$ in the next time step:

$$
r_g(s_t, a_t) \triangleq (1 - \gamma)p(s_{t+1} = g \mid s_t, a_t)
$$

where $\gamma$ is the discount factor.

### Discounted State Visitation Distribution

Conditioned on a policy $\pi$ and a goal $g$, the discounted state visitation distribution is defined as:

$$
p_\gamma^{\pi(\cdot|\cdot, g)}(s) \triangleq (1 - \gamma) \sum_{t=0}^{\infty} \gamma^t p_t^{\pi}(s)
$$

where $p_t^{\pi}(s)$ is the probability that policy $\pi$ visits state $s$ at step $t$.

### Goal-Conditioned Q-Function

Under this formulation, the Q-function is equivalent to the probability of visiting the goal $g$ in the future, starting from $(s,a)$:

$$
Q_g^\pi(s, a) \triangleq p_\gamma^{\pi(\cdot|\cdot, g)}(g \mid s, a)
$$

This allows us to train the critic as a classifier using contrastive learning rather than regression.

### The Objective

The general goal is to maximize the expected cumulative discounted reward over the distribution of initial states $p_0(s_0)$ and goals $p_g(g)$:

$$
\max_{\pi} \mathbb{E}_{s_0 \sim p_0, g \sim p_g, \pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_g(s_t, a_t) \right]
$$

By plugging in our probabilistic definition of the reward $r_g(s_t, a_t)$, the objective becomes maximizing the total discounted probability mass of visiting the commanded goal:

$$
\max_{\pi} \mathbb{E}_{s_0 \sim p_0, g \sim p_g, \pi} \left[ \sum_{t=0}^{\infty} \gamma^t (1 - \gamma) p(s_{t+1} = g \mid s_t, a_t) \right]
$$

### InfoNCE Loss

To optimize the Q-function (critic) as a classifier, we employ the InfoNCE loss. Let the critic score be defined as the negative $L_2$ distance between the state-action embedding $\phi(s,a)$ and the goal embedding $\psi(g)$, denoted as $f(s, a, g) = -\|\phi(s, a) - \psi(g)\|_2$. Given a positive goal $g^+$ (a future state from the same trajectory) and a set of $K$ negative goals $\{g^-_j\}_{j=1}^K$ (sampled from different trajectories), the loss is:

$$
\mathcal{L}_{\text{InfoNCE}} = \mathbb{E}_{\mathcal{D}} \left[ -\log \frac{\exp(f(s, a, g^+))}{\exp(f(s, a, g^+)) + \sum_{j=1}^{K} \exp(f(s, a, g^{-_j}))} \right]
$$

---

### Jan12 - Jan18 Updates:

1. implemented contrastive reinforcement learning in pytorch.

2. studied the metaworld environment in detail based on its official documentation.

3. investigated metaworld by reading through the source code, see [TASK_GOAL_ANALYSIS](metaworld/TASK_GOAL_ANALYSIS.md).

4. clarified what information is available at environment initialization, including the format and shape of observations, rewards, goals, and internal states, see [METAWORLD_ENVIRONMENT_ANALYSIS](metaworld/METAWORLD_ENVIRONMENT_ANALYSIS.md).

5. a preliminary environment wrapper to redefine the reward, customize reset behavior, and explicitly define desired_goal and critic_goal.

6. debugged and verified that the critic encoders and the environment wrapper work correctly with the existing cka-rl codebase for continual learning, including baseline methods such as cka-rl, packnet, and self-component.

7. revised the training pipeline to ensure that the critic is not reinitialized when a new task arrives, and that learned information is preserved across tasks.

8. instead of using the stable-baselines replay buffer, implemented a new buffer.py from scratch to support goal sampling from future states of a given state–action pair, while also recording trajectory IDs and related metadata.

9. debugged the full pipeline and tested it on a single task, specifically task 3 in metaworld.

10. collected preliminary observations from these experiments.

---
### Jan 20 Meeting:
- Based on the definition $r_g(s_t, a_t) = (1-\gamma)p(s_{t+1}=g|s_t, a_t)$, the reward signal for 99% of the trajectory is effectively **zero**.
- We can rewrite the objective (ignoring the scaling constant $1-\gamma$ for clarity, as it doesn't change the maximization) as:

$$ \max_\pi \mathbb{E}_{\tau \sim \pi, g \sim p_g} \left[ \sum_{t=0}^{\infty} \gamma^t \cdot \mathbf{1}(s_{t+1} = g) \right] $$
1.  **The Trajectory:** the agent generates a path: $s_0, s_1, s_2, \dots, s_{100}$.
2.  **The Sparse Signal:**
    *   At $t=0$, is $s_1$ the goal? No. $\to \mathbf{1}(\dots) = 0$.
    *   At $t=1$, is $s_2$ the goal? No. $\to \mathbf{1}(\dots) = 0$.
    *   ...
    *   At $t=k$, is $s_{k+1}$ the goal? **YES.** $\to \mathbf{1}(\dots) = 1$.
3.  **The Sum:** The summation collapses to:
    $$
    0 + 0 + \dots + (\gamma^k \cdot 1) + (\gamma^{k+1} \cdot 1) + \dots
    $$
    *(Assuming that once the agent reaches the goal, it stays there).*
This effectively means the objective is to **minimize $k$** (the time it takes to reach the goal).
Since $\gamma < 1$ (e.g., 0.99), the faster you reach the goal (smaller $k$), the larger the term $\gamma^k$ becomes.

*   If you reach the goal in 5 steps: Total Value $\approx 0.99^5 \approx 0.95$.
*   If you reach the goal in 100 steps: Total Value $\approx 0.99^{100} \approx 0.36$.
*   If you never reach the goal: Total Value = $0$.

So, maximizing this sparse expectation is mathematically equivalent to **"Reaching the goal as fast as possible."**

---

### Jan 20 Updates:
1. Notice the off-policy nature of the Actor.
   - Rollout collection/behavior policy: goal is the environment's goal.
   - Training updates/target policy: goal uses hindsight relabeling and samples from the replay buffer.
2. The masking detail in their setup for the infoNCE loss function
   - Only one positive sample.
   - No distinction between `(s, a, g)` from the same trajectory vs. different trajectories in the replay buffer.
3. Restructure the replay buffer.
4. Sampling diversity
   - Sample transitions `(s, a, s', g')` from as many different trajectories as possible.
5. Sliced goal clarification
   - The sliced goal is the cube position, not arm position or distance.
   - If the cube never moves, the future-state goal is nearly constant across that trajectory, so HER won't provide varied goals; learning can stall unless exploration moves the cube (or other trajectories do).
6. Evaluation metric
   - They report "time at goal" rather than the success condition.
7. Their UTD ratio is high.



---
### Jan 21 Updates:
1. For the manipulation task in that paper, specifically arm-push,  the environment has one goal designating the object's final coordinates, let's denote it as g_env. During the rollouts (collecting the experiences), actor always see observation + (g_env + some random noise); during the training of the actor network, they sample transitions from replay buffer, they use hindsight relabeling to change the goal to make it align with the ground truth future state.

2. Notice that when they report their results, they show their main figure y-axis as “time at goal” rather than the success rate of the tasks. For the arm-push task, the environment goal is the object position of the cube. They initialize their robotic arm’s position to be very close to the cube, such that during the exploration phase the hand can easily touch the object and hence change the object’s position.
   
   If the actor never touches or moves the cube, the HER goals will stay constant for those trajectories, and learning can stall until some exploration episodes do move the cube. They make the robotic arm close to the cube, so during exploration the arm can actually touch it, thereby changing the object position, and these changes contribute to the learning signals in the critics’ encoders. Hence, for their experiment, they can simply set the goal as the object’s coordinates without further engineering, such as adding another goal like the distance between the arm and the object.
   
   For Meta-World, it is a different story. During the exploration phase, the arm can never touch the cube, so nothing ever changes in any future state’s relabeling, resulting in no learning signals (given that the goal is a subset of the state, and here it is just the indices of the object’s position).
   
   We need to engineer the goal a bit, or we can also position our arm close to the cube, but I do not want to do that.
   
   To resolve the issue of the arm never touching the cube during the exploration phase, I change the whole actor backbone to see if a stronger actor can eventually touch the cube.
   But more importantly, as I suggested, I think it is about the goal engineering.
---
### Jan 22 Updates:
1. Regarding the UTD ratio:
  - They do one data collection step (an “actor step”) that generates num_envs * unroll_length transitions, and then run a fixed number of SGD updates on batches from the replay buffer. That’s where the UTD comes from.
  - With the defaults:
    - num_envs=512
    - unroll_length=62
    - num_sgd_batches_per_training_step=800
    - Env steps per training step = 512*62 = 31,744.
  - Updates per training step = 800.UTD is about 800 / 31,744 which is 0.0252; about 1 update per 40 environment transitions.
2. Regarding the Critic Training: 
  - During training, each sampled trajectory is relabeled inside flatten_crl_fn: it picks a future state from the same episode, extracts an achieved-goal slice from that future state, and replaces the observation’s goal with that achieved goal. This is the goal stored in transitions.observation used for training.The critic uses transitions.observation[:, args.obs_dim:] as its goal input, so it sees the relabeled goal, not the original desired goal from the env.The actor explicitly rebuilds an observation using future_state (same relabeled future), so it is conditioned on the same goal distribution as the critic.
3. ToDos:
  - Goal engineering for the metaworld task 3 (add another goal about [armposition - bloack]) so that HER relabeling can be changed.
  - Try run multiple environments in parallel as well (perhaps less priority).
  - Read their paper on Metaworld.

---