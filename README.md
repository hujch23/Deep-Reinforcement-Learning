# What can i say, just do it!


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/hujch23/Deep-Reinforcement-Learning/issues)

## 📚 目录

- [基础概念](#基础概念)
- [核心算法](#核心算法)
  - [动态规划vs蒙特卡洛vs时序差分](##动态规划vs蒙特卡洛vs时序差分)
  - [Q-learning和Sarsa](##Q-learning和Sarsa)
  - [DQN算法及变种](##DQN及变种)
  - [策略梯度方法](##策略梯度方法)
  - [actor-critic方法及变种](##actor-critic方法)
  - [PPO算法及变种](##PPO方法及变种)
  - [Molde-based_SOTA方法](##Molde-based_SOTA方法)
- [调参技巧](#调参技巧)
- [实际应用](#实际应用)



## 📖 1. 基础概念  

### 1.1 强化学习的基本组成部分有哪些？基本特征有哪些？
  - Agent（智能体） Environment（环境） State（状态）Action（动作） Reward（奖励） Policy（策略）
  -   有试错探索过程，即需要通过探索环境来获取对当前环境的理解
  -   强化学习中的智能体会从环境中获得延迟奖励
  -   强化学习的训练过程中时间非常重要，因为数据都是时间关联的，而不是像监督学习中的数据大部分是满足独立同分布的
  -   强化学习中智能体的动作会影响它从环境中得到的反馈

### 1.2 解释什么是回报（Return）和价值函数（Value Function）？  
  - 回报：从某一时刻开始，未来所有奖励的折扣总和  
  - 价值函数：从某个状态开始，遵循特定策略能够获得的期望回报  

### 1.3 全部可观测（full observability）、完全可观测（fully observed）和部分可观测（partially observed）?
  - 当智能体的状态与环境的状态等价时，我们就称这个环境是全部可观测的
  - 当智能体能够观察到环境的所有状态时，我们称这个环境是完全可观测的
  - 一般智能体不能观察到环境的所有状态时，我们称这个环境是部分可观测的

### 1.4 强化学习的使用场景有哪些呢？
  - 7个字总结就是“多序列决策问题”，或者说是对应的模型未知，需要通过学习逐渐逼近真实模型的问题。并且当前的动作会影响环境的状态，即具有马尔可夫性的问题。同时应满足所有状态是可重复到达的条件，即满足可学习条件

### 1.5 强化学习相对于监督学习为什么训练过程会更加困难？
  - 处理的大多是序列数据，其很难像监督学习的样本一样满足独立同分布条件
  - 有奖励的延迟，即智能体的动作作用在环境中时，环境对于智能体状态的奖励存在延迟，使得反馈不实时
  - 监督学习有正确的标签，模型可以通过标签修正自己的预测来更新模型，而强化学习相当于一个“试错”的过程，其完全根据环境的“反馈”更新对自己最有利的动作

### 1.6 状态和观测有什么关系？
  - 状态是对环境的完整描述，不会隐藏环境信息。观测是对状态的部分描述，可能会遗漏一些信息。在深度强化学习中，我们几乎总是用同一个实值向量、矩阵或者更高阶的张量来表示状态和观测

### 1.7 根据强化学习智能体的不同，我们可以将其分为哪几类？
  - 基于价值的智能体。显式学习的是价值函数，隐式地学习智能体的策略。因为这个策略是从学到的价值函数里面推算出来的
  - 基于策略的智能体。其直接学习策略，即直接给智能体一个状态，它就会输出对应动作的概率。当然在基于策略的智能体里面并没有去学习智能体的价值函数
  - 另外还有一种智能体，它把以上两者结合。把基于价值和基于策略的智能体结合起来就有了演员-评论员智能体。这一类智能体通过学习策略函数和价值函数以及两者的交互得到更佳的状态

### 1.8 强化学习、监督学习和无监督学习三者有什么区别呢？
  - 首先强化学习和无监督学习是不需要有标签样本的，而监督学习需要许多有标签样本来进行模型的构建训练。其次对于强化学习与无监督学习，无监督学习直接基于给定的数据进行建模，寻找数据或特征中隐藏的结构，一般对应聚类问题；强化学习需要通过延迟奖励学习策略来得到模型与目标的距离，这个距离可以通过奖励函数进行定量判断，这里我们可以将奖励函数视为正确目标的一个稀疏、延迟形式。另外，强化学习处理的多是序列数据，样本之间通常具有强相关性，但其很难像监督学习的样本一样满足独立同分布条件

### 1.9 基于策略迭代和基于价值迭代的强化学习方法有什么区别？
  - 基于策略迭代的强化学习方法，智能体会制定一套动作策略，即确定在给定状态下需要采取何种动作，并根据该策略进行操作。强化学习算法直接对策略进行优化，使得制定的策略能够获得最大的奖励；基于价值迭代的强化学习方法，智能体不需要制定显式的策略，它维护一个价值表格或价值函数，并通过这个价值表格或价值函数来选取价值最大的动作
  - 基于价值迭代的方法只能应用在离散的环境下，例如围棋或某些游戏领域，对于行为集合规模庞大或是动作连续的场景，如机器人控制领域，其很难学习到较好的结果（此时基于策略迭代的方法能够根据设定的策略来选择连续的动作)
  - 基于价值迭代的强化学习算法有 Q-learning、Sarsa 等，基于策略迭代的强化学习算法有策略梯度算法等
  - 此外，演员-评论员算法同时使用策略和价值评估来做出决策。其中，智能体会根据策略做出动作，而价值函数会对做出的动作给出价值，这样可以在原有的策略梯度算法的基础上加速学习过程，从而取得更好的效果

### 1.10 Model-based 和 Model-free、 On Policy 和 Off Policy、Online and Offline、Value-based 和 Policy-based
  - Online和Offline，主要的区别在于智能体训练时是否实时与环境进行交互。Online RL 依赖于实时交互，而 Offline RL 则依赖于预先收集的数据（根据数据稀缺程度选择）
  - On-policy 和Off-policy皆属于Online RL，主要的区别在于是否使用与当前策略相同的数据来进行学习。On-policy 仅使用当前策略产生的数据来更新策略，而 Off-policy 可以使用其他策略生成的数据来学习
  - 在强化学习中，所谓的“模型”一般都指的是环境的模型，即环境的动态模型，通常包含两部分：一是状态转移（state transition）函数，二是奖励（reward）函数

### 1.11 强化学习中所谓的损失函数与深度学习中的损失函数有什么区别呢？
  - 深度学习中的损失函数的目的是使预测值和真实值之间的差距尽可能小，而强化学习中的损失函数的目的是使总奖励的期望尽可能大

### 1.12 马尔科夫决策核心词汇
  - 马尔可夫性质（Markov property，MP）：如果某一个过程未来的状态与过去的状态无关，只由现在的状态决定，那么其具有马尔可夫性质。换句话说，一个状态的下一个状态只取决于它的当前状态，而与它当前状态之前的状态都没有关系
  
  - 马尔可夫链（Markov chain）： 概率论和数理统计中具有马尔可夫性质且存在于离散的指数集（index set）和状态空间（state space）内的随机过程（stochastic process）
  - 状态转移矩阵（state transition matrix）：状态转移矩阵类似于条件概率（conditional probability），其表示当智能体到达某状态后，到达其他所有状态的概率。矩阵的每一行描述的是从某节点到达所有其他节点的概率
    
  - 马尔可夫奖励过程（Markov reward process，MRP）： 本质是马尔可夫链加上一个奖励函数。在马尔可夫奖励过程中，状态转移矩阵和它的状态都与马尔可夫链的一样，只多了一个奖励函数。奖励函数是一个期望，即在某一个状态可以获得多大的奖励
  - 范围（horizon）：定义了同一个回合（episode）或者一个完整轨迹的长度，它是由有限个步数决定的
  - 回报（return）：把奖励进行折扣（discounted）求和
  - 贝尔曼方程（Bellman equation）：其定义了当前状态与未来状态的迭代关系，表示当前状态的价值函数可以通过下个状态的价值函数来计算。贝尔曼方程因其提出者、动态规划创始人理查德 $\cdot$ 贝尔曼（Richard Bellman）而得名，同时也被叫作“动态规划方程”
  - 蒙特卡洛算法（Monte Carlo algorithm，MC algorithm）： 在马尔可夫奖励过程中，从特定状态开始生成多条轨迹，计算每条轨迹的折扣总奖励，最后取平均值即为该状态的价值函数估计
  - 动态规划算法（dynamic programming，DP）： 其可用来计算价值函数的值。通过一直迭代对应的贝尔曼方程，最后使其收敛。当最后更新的状态与上一个状态差距不大的时候，动态规划算法的更新就可以停止
  - Q函数（Q-function）： 其定义的是某一个状态和某一个动作所对应的有可能得到的回报的期望
  - 马尔可夫决策过程中的预测问题：即策略评估问题，给定一个马尔可夫决策过程以及一个策略 $\pi$ ，计算它的策略函数，即每个状态的价值函数值是多少。其可以通过动态规划算法解决
  - 马尔可夫决策过程中的控制问题：即寻找一个最佳策略，其输入是马尔可夫决策过程，输出是最佳价值函数（optimal value function）以及最佳策略（optimal policy）。其可以通过动态规划算法解决
  - 最佳价值函数：搜索一种策略 $\pi$ ，使每个状态的价值最大，![equation](https://latex.codecogs.com/svg.latex?V^*) 就是到达每一个状态的极大值。在极大值中，我们得到的策略是最佳策略。最佳策略使得每个状态的价值函数都取得最大值。所以当我们说某一个马尔可夫决策过程的环境可解时，其实就是我们可以得到一个最佳价值函数

### 1.13 为什么在马尔可夫奖励过程中需要有折扣因子？
  - 数学上的必要性：确保无限时间序列的奖励之和能够收敛、避免累积奖励变成无穷大
  - 反映现实决策偏好：体现"现在的奖励比将来的更有价值"这一现实思维
  - 处理不确定性：通过折扣因子降低远期奖励的权重，更合理地应对不确定性

### 1.14 计算贝尔曼方程的常见方法有哪些，它们有什么区别？
  - 蒙特卡洛方法：可用来计算价值函数的值。
  - 动态规划方法：可用来计算价值函数的值。通过一直迭代对应的贝尔曼方程，最后使其收敛。当最后更新的状态与上一个状态区别不大的时候，通常是小于一个阈值 $\gamma$ 时，更新就可以停止
  - 以上两者的结合方法：我们也可以使用时序差分学习方法，其为动态规划方法和蒙特卡洛方法的结合
  - 具体见2.1 

### 1.15 马尔可夫奖励过程与马尔可夫决策过程的区别是什么？
  - MRP: 状态转移是自动的，由固定的转移概率决定
  - MDP: 状态转移受智能体选择的动作影响，P(s'|s,a)依赖于动作a
  - MRP: R(s)只与状态有关
  - MDP: R(s,a,s')与状态和动作都相关
  - MDP是MRP的扩展，增加了动作选择的能力，使系统从被动观察变为主动决策。

### 1.16 马尔可夫过程是什么？马尔可夫决策过程又是什么？其中马尔可夫最重要的性质是什么呢？
  - 马尔可夫过程是一个二元组 $<S,P>$ ， $S$ 为状态集合， $P$ 为状态转移函数
  - 马尔可夫决策过程是一个五元组 $<S,P,A,R,\gamma>$， 其中 $R$ 表示从 $S$ 到 $S'$ 能够获得的奖励期望， $\gamma$ 为折扣因子， $A$ 为动作集合
  - 马尔可夫最重要的性质是下一个状态只与当前状态有关，与之前的状态无关

### 1.17 如果数据流不具备马尔可夫性质怎么办？应该如何处理？
  - 如果不具备马尔可夫性，即下一个状态与之前的状态也有关，若仅用当前的状态来求解决策过程，势必导致决策的泛化能力变差。为了解决这个问题，可以利用循环神经网络对历史信息建模，获得包含历史信息的状态表征，表征过程也可以使用注意力机制等手段，最后在表征状态空间求解马尔可夫决策过程问题。

### 1.18 写出基于状态价值函数的贝尔曼方程以及基于动作价值函数的贝尔曼方程

  - 基于状态价值函数的贝尔曼方程:
```math
V_{\pi}(s) = \sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r(s,a)+\gamma V_{\pi}(s')]
```
  - 基于动作价值函数的贝尔曼方程:
```math
Q_{\pi}(s,a)=\sum_{s',r}p(s',r|s,a)r(s',a)+\gamma V_{\pi}(s')
```

### 1.19 一般怎么求解马尔可夫决策过程？
  - 我们求解马尔可夫决策过程时，可以直接求解贝尔曼方程或动态规划方程。但是贝尔曼方程很难求解且计算复杂度较高，所以可以使用动态规划、蒙特卡洛以及时序差分等方法求解

## 🏆 2. 核心算法  

### 2.1 动态规划vs蒙特卡洛vs时序差分

#### 2.1.1 📊 方法概览

| 特性 | 动态规划 (DP) | 蒙特卡洛 (MC) | 时序差分 (TD) |
|------|--------------|--------------|--------------|
| 需要环境模型 | ✅ 需要完整模型 | ❌ 不需要 | ❌ 不需要 |
| 学习方式 | 自举学习 | 采样学习 | 自举+采样 |
| 更新时机 | 每步更新 | 回合结束 | 每步更新 |
| 计算复杂度 | 较高 | 中等 | 较低 |
| 收敛速度 | 快 | 慢 | 中等 |
| 特点 | 理论完备但计算昂贵	 | 无偏但方差大 | 平衡偏差和方差，实践最常用 |

动态规划有两种主要方法：策略迭代（Policy Iteration）和值迭代（Value Iteration）。让我详细解释：
- 策略迭代分两步走：首先评估当前策略的好坏（计算在当前策略下每个状态的价值），然后基于这个评估来改进策略（选择能获得更高价值的动作）。这两步不断重复，直到策略不能再改进为止。就像不断调整下棋策略，每次都先评估当前策略的效果，再做出改进。
- 值迭代则更直接：它直接计算每个状态的最优价值，不用分成评估和改进两步。它假设每次都选择能带来最大价值的动作，不断更新状态的价值，直到价值稳定下来。最后根据这些价值选出最优策略。这就像直接计算每个棋局的最佳价值，最后根据这些价值决定怎么走棋。

#### 2.1.2 🔍 主要区别

1. **环境模型要求**  DP：需要完整的环境模型（状态转移概率和奖励函数）、MC：只需要能够采样经验 、TD：只需要能够采样经验
2. **学习特点**  DP：基于状态转移和奖励的确定性计算、MC：基于完整回合的实际回报、TD：结合DP的自举思想和MC的采样思想
3. **更新机制**  DP：系统地更新所有状态、MC：回合结束后才能更新、TD：可以在线学习，即时更新

#### 2.1.3💡 共同点

1. **基本框架** 都遵循贝尔曼方程、都是迭代式的价值更新方法、都能用于策略评估和策略改进
2. **目标** 都致力于估计价值函数、都能用于寻找最优策略、都能收敛到最优解（在适当条件下）

### 2.2 Q-learning 和 Sarsa

#### 2.2.1 Q-learning 和 Sarsa区别？
  - 首先，Q学习是异策略的时序差分学习方法，而 Sarsa 算法是同策略的时序差分学习方法
  - 其次，Sarsa算法在更新Q表格的时候所用到的 $a'$ 是获取下一个Q值时一定会执行的动作。这个动作有可能是用 $\varepsilon$-贪心方法采样出来的，也有可能是 $\mathrm{max}_Q$ 对应的动作，甚至是随机动作。
  - 但是Q学习在更新Q表格的时候所用到的Q值 $Q(S',a')$ 对应的动作不一定是下一步会执行的动作，因为下一步实际会执行的动作可能是因为进一步的探索而得到的。Q学习默认的动作不是通过行为策略来选取的，它默认 $a'$ 为最佳策略对应的动作，所以Q学习算法在更新的时候，不需要传入 $a'$ ，即 $a_{t+1}$ 。
  - 更新公式的对比（区别只在目标计算部分），Sarsa算法的公式：![equation](https://latex.codecogs.com/svg.latex?r_{t+1}+\gamma%20Q(s_{t+1},%20a_{t+1}))， Q学习算法的公式：![equation](https://latex.codecogs.com/svg.latex?r_{t+1}+\gamma%20\underset{a}{\max}%20Q(s_{t+1},%20a))， 总结起来，Sarsa算法实际上是用固有的策略产生 (S, A, R, S', A') 这一条轨迹，然后使用 Q(s<sub>t+1</sub>, a<sub>t+1</sub>) 更新原本的Q值 Q(s<sub>t</sub>, a<sub>t</sub>)。但是Q学习算法并不需要知道实际上选择的动作，它默认下一个动作就是Q值最大的那个动作。所以Sarsa算法的动作通常会更加“保守胆小”，而对应的Q学习算法的动作会更加“莽撞激进”。

#### 2.2.2 同策略和异策略的区别是什么？ 
  - Sarsa算法就是一个典型的同策略算法，它只用一个 $\pi$ ，为了兼顾探索和开发，它在训练的时候会显得有点儿“胆小怕事”。它在解决悬崖寻路问题的时候，会尽可能地远离悬崖边，确保哪怕自己不小心向未知区域探索了一些，也还是处在安全区域内，不至于掉入悬崖中
  - Q学习算法是一个比较典型的异策略算法，它有目标策略（target policy），用 $\pi$ 来表示。此外还有行为策略（behavior policy），用 $\mu$ 来表示。它分离了目标策略与行为策略，使得其可以大胆地用行为策略探索得到的经验轨迹来优化目标策略。这样智能体就更有可能探索到最优的策略
  - 比较Q学习算法和Sarsa算法的更新公式可以发现，Sarsa算法并没有选取最大值的操作。因此，Q学习算法是非常激进的，其希望每一步都获得最大的奖励；Sarsa算法则相对来说偏保守，会选择一条相对安全的迭代路线

#### 补充知识：
**N步Q-learning与重要性采样的数学推导**

**1. N步回报的定义**

在n步Q-learning中，n步回报 $G_t^{(n)}$ 定义为：

$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{n-1}R_{t+n} + \gamma^n \max_a Q(S_{t+n}, a)$

其中：
- $R_t$ 是在时间步t获得的即时奖励
- $\gamma$ 是折扣因子
- $Q(S_{t+n}, a)$ 是状态-动作值函数
- $n$ 是步数

**2. 重要性采样比率**

当行为策略 $\mu$ 与目标策略 $\pi$ 不同时，重要性采样比率 $\rho_t$ 定义为：

$\rho_t = \frac{\pi(A_t|S_t)}{\mu(A_t|S_t)}$

其中：
- $\pi(A_t|S_t)$ 是目标策略选择动作 $A_t$ 的概率
- $\mu(A_t|S_t)$ 是行为策略选择动作 $A_t$ 的概率

**3. n步累积重要性权重**

对于n步序列，累积重要性权重 $\rho_{t:t+n-1}$ 为：

$\rho_{t:t+n-1} = \prod_{k=t}^{t+n-1} \rho_k = \prod_{k=t}^{t+n-1} \frac{\pi(A_k|S_k)}{\mu(A_k|S_k)}$

**4. 带重要性采样的n步Q-learning更新**

Q值更新公式：

$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \rho_{t:t+n-1}[G_t^{(n)} - Q(S_t,A_t)]$

展开后：

$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \prod_{k=t}^{t+n-1} \frac{\pi(A_k|S_k)}{\mu(A_k|S_k)} [R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1}R_{t+n} + \gamma^n \max_a Q(S_{t+n}, a) - Q(S_t,A_t)]$

**5. 方差分析**

重要性采样比率的方差：

$Var(\rho_{t:t+n-1}) = E[(\prod_{k=t}^{t+n-1} \frac{\pi(A_k|S_k)}{\mu(A_k|S_k)})^2] - (E[\prod_{k=t}^{t+n-1} \frac{\pi(A_k|S_k)}{\mu(A_k|S_k)}])^2$

**6. 截断重要性采样**

为控制方差，使用截断重要性采样：

$\rho_{t:t+n-1}^c = \min(c, \prod_{k=t}^{t+n-1} \frac{\pi(A_k|S_k)}{\mu(A_k|S_k)})$

其中 $c$ 是截断阈值。

class NStepQLearningMath:  

    def __init__(self, n_actions, n_steps=3, epsilon=0.1, alpha=0.5, gamma=0.95):  
        self.n_actions = n_actions  
        self.n_steps = n_steps  
        self.epsilon = epsilon  
        self.alpha = alpha  
        self.gamma = gamma  
        self.Q = defaultdict(lambda: np.zeros(n_actions))  
        
    def calculate_importance_ratio(self, state, action):  
        """  
        计算单步重要性采样比率  
        ρ_t = π(A_t|S_t) / μ(A_t|S_t)  
        """  
        # 目标策略概率 (贪婪策略)  
        target_prob = 1.0 if action == np.argmax(self.Q[state]) else 0.0  
        
        # 行为策略概率 (ε-贪婪)  
        if action == np.argmax(self.Q[state]):  
            behavior_prob = 1 - self.epsilon + self.epsilon / self.n_actions  
        else:  
            behavior_prob = self.epsilon / self.n_actions  
            
        return target_prob / (behavior_prob + 1e-8)  
    
    def calculate_n_step_return(self, rewards, next_state=None):  
        """  
        计算n步回报  
        G_t^(n) = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n max_a Q(S_{t+n},a)  
        """  
        G = 0  
        for i, r in enumerate(rewards):  
            G += self.gamma**i * r  
            
        if next_state is not None:  
            G += self.gamma**len(rewards) * np.max(self.Q[next_state])  
            
        return G  
    
    def calculate_cumulative_importance_weight(self, importance_ratios):  
        """  
        计算累积重要性权重  
        ρ_{t:t+n-1} = ∏_{k=t}^{t+n-1} ρ_k  
        """  
        return np.prod(importance_ratios)  
    
    def truncated_importance_sampling(self, ratio, c=10.0):  
        """  
        截断重要性采样  
        ρ^c = min(c, ρ)  
        """  
        return min(ratio, c)  
    
    def update_q_value(self, state, action, n_step_return, importance_weight):  
        """  
        Q值更新  
        Q(S_t,A_t) ← Q(S_t,A_t) + α * ρ_{t:t+n-1} * [G_t^(n) - Q(S_t,A_t)]  
        """  
        current_q = self.Q[state][action]  
        td_error = n_step_return - current_q  
        self.Q[state][action] += self.alpha * importance_weight * td_error  
        
        return td_error  
    
    def calculate_variance(self, importance_ratios):  
        """  
        计算重要性采样比率的方差  
        Var(ρ) = E[ρ^2] - (E[ρ])^2  
        """  
        mean = np.mean(importance_ratios)  
        mean_square = np.mean([r**2 for r in importance_ratios])  
        return mean_square - mean**2  
    
    def adaptive_learning_rate(self, importance_weight):  
        """  
        基于重要性权重调整学习率  
        α' = α / (1 + ||ρ||)  
        """  
        return self.alpha / (1 + np.abs(importance_weight))  

# 示例使用  
def example_calculation():  
    """展示数学计算示例"""  
    agent = NStepQLearningMath(n_actions=4)  
    
    # 假设的状态和动作  
    state = (0, 0)  
    action = 1  
    rewards = [1, 0.5, 2]  
    next_state = (1, 0)  
    
    # 1. 计算单步重要性比率  
    ratio = agent.calculate_importance_ratio(state, action)  
    print(f"Single-step importance ratio: {ratio:.4f}")  
    
    # 2. 计算n步回报  
    n_step_return = agent.calculate_n_step_return(rewards, next_state)  
    print(f"N-step return: {n_step_return:.4f}")  
    
    # 3. 计算累积重要性权重  
    ratios = [agent.calculate_importance_ratio(state, action) for _ in range(3)]  
    cum_weight = agent.calculate_cumulative_importance_weight(ratios)  
    print(f"Cumulative importance weight: {cum_weight:.4f}")  
    
    # 4. 计算截断重要性采样  
    truncated_weight = agent.truncated_importance_sampling(cum_weight)  
    print(f"Truncated importance weight: {truncated_weight:.4f}")  
    
    # 5. 更新Q值  
    td_error = agent.update_q_value(state, action, n_step_return, truncated_weight)  
    print(f"TD error: {td_error:.4f}")  
    
    # 6. 计算方差  
    variance = agent.calculate_variance(ratios)  
    print(f"Importance sampling variance: {variance:.4f}")  
    
    # 7. 自适应学习率  
    adaptive_alpha = agent.adaptive_learning_rate(truncated_weight)  
    print(f"Adaptive learning rate: {adaptive_alpha:.4f}")  

if __name__ == "__main__":  
    example_calculation()

### 2.3 DQN算法及变种







