# 求职RL算法工程师必背知识点

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/hujch23/Deep-Reinforcement-Learning/issues)

## 📚 目录

- [基础概念](#基础概念)
- [核心算法](#核心算法)


## 🚀 1. 基础概念  

### 1.1 强化学习的基本组成部分有哪些？基本特征有哪些？
  - Agent（智能体）
  - Environment（环境）
  -  State（状态）
  -   Action（动作）
  -   Reward（奖励）
  -   Policy（策略）
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
  - 回报（return）：把奖励进行折扣（discounted），然后获得的对应的奖励
  - 贝尔曼方程（Bellman equation）：其定义了当前状态与未来状态的迭代关系，表示当前状态的价值函数可以通过下个状态的价值函数来计算。贝尔曼方程因其提出者、动态规划创始人理查德 $\cdot$ 贝尔曼（Richard Bellman）而得名，同时也被叫作“动态规划方程”。贝尔曼方程即 $V(s)=R(s)+ \gamma \sum_{s' \in S}P(s'|s)V(s')$ ，特别地，其矩阵形式为 $\mathrm{V}=\mathrm{R}+\gamma \mathrm{PV}$
  - 蒙特卡洛算法（Monte Carlo algorithm，MC algorithm）： 可用来计算价值函数的值。使用本节中小船的例子，当得到一个马尔可夫奖励过程后，我们可以从某一个状态开始，把小船放到水中，让它随波流动，这样就会产生一个轨迹，从而得到一个折扣后的奖励 $g$ 。当积累该奖励到一定数量后，用它直接除以轨迹数量，就会得到其价值函数的值
  - 动态规划算法（dynamic programming，DP）： 其可用来计算价值函数的值。通过一直迭代对应的贝尔曼方程，最后使其收敛。当最后更新的状态与上一个状态差距不大的时候，动态规划算法的更新就可以停止
  - Q函数（Q-function）： 其定义的是某一个状态和某一个动作所对应的有可能得到的回报的期望
  - 马尔可夫决策过程中的预测问题：即策略评估问题，给定一个马尔可夫决策过程以及一个策略 $\pi$ ，计算它的策略函数，即每个状态的价值函数值是多少。其可以通过动态规划算法解决
  - 马尔可夫决策过程中的控制问题：即寻找一个最佳策略，其输入是马尔可夫决策过程，输出是最佳价值函数（optimal value function）以及最佳策略（optimal policy）。其可以通过动态规划算法解决
  - 最佳价值函数：搜索一种策略 $\pi$ ，使每个状态的价值最大，$V^*$ 就是到达每一个状态的极大值。在极大值中，我们得到的策略是最佳策略。最佳策略使得每个状态的价值函数都取得最大值。所以当我们说某一个马尔可夫决策过程的环境可解时，其实就是我们可以得到一个最佳价值函数

### 1.13 为什么在马尔可夫奖励过程中需要有折扣因子？
  - 首先，是有些马尔可夫过程是环状的，它并没有终点，所以我们想避免无穷的奖励
  - 另外，我们想把不确定性也表示出来，希望尽可能快地得到奖励，而不是在未来的某个时刻得到奖励
  - 接上一点，如果这个奖励是有实际价值的，我们可能更希望立刻就得到奖励，而不是后面才可以得到奖励
  - 还有，在有些时候，折扣因子也可以设为0。当它被设为0后，我们就只关注它当前的奖励。我们也可以把它设为1，设为1表示未来获得的奖励与当前获得的奖励是一样的
所以，折扣因子可以作为强化学习智能体的一个超参数进行调整，然后就会得到不同行为的智能体

### 1.14 计算贝尔曼方程的常见方法有哪些，它们有什么区别？
  - 蒙特卡洛方法：可用来计算价值函数的值。以本书中的小船示例为例，当得到一个马尔可夫奖励过程后，我们可以从某一个状态开始，把小船放到水中，让它“随波逐流”，这样就会产生一条轨迹，从而得到一个折扣后的奖励 $g$ 。当积累该奖励到一定数量后，直接除以轨迹数量，就会得到其价值函数的值
  - 动态规划方法：可用来计算价值函数的值。通过一直迭代对应的贝尔曼方程，最后使其收敛。当最后更新的状态与上一个状态区别不大的时候，通常是小于一个阈值 $\gamma$ 时，更新就可以停止
  - 以上两者的结合方法：我们也可以使用时序差分学习方法，其为动态规划方法和蒙特卡洛方法的结合

### 1.15 马尔可夫奖励过程与马尔可夫决策过程的区别是什么？
  - 相对于马尔可夫奖励过程，马尔可夫决策过程多了一个决策过程，其他的定义与马尔可夫奖励过程是类似的。由于多了一个决策，多了一个动作，因此状态转移也多了一个条件，即执行一个动作，导致未来状态的变化，其不仅依赖于当前的状态，也依赖于在当前状态下智能体采取的动作决定的状态变化。对于价值函数，它也多了一个条件，多了一个当前的动作，即当前状态以及采取的动作会决定当前可能得到的奖励的多少。
  - 另外，两者之间是有转换关系的。具体来说，已知一个马尔可夫决策过程以及一个策略 $\pi$ 时，我们可以把马尔可夫决策过程转换成马尔可夫奖励过程。在马尔可夫决策过程中，状态的转移函数 $P(s'|s,a)$ 是基于它的当前状态和当前动作的，因为我们现在已知策略函数，即在每一个状态，我们知道其采取每一个动作的概率，所以我们就可以直接把这个动作进行加和，就可以得到对于马尔可夫奖励过程的一个转移概率。同样地，对于奖励，我们可以把动作去掉，这样就会得到一个类似于马尔可夫奖励过程的奖励

### 1.16 马尔可夫过程是什么？马尔可夫决策过程又是什么？其中马尔可夫最重要的性质是什么呢？
  - 马尔可夫过程是一个二元组 $<S,P>$ ， $S$ 为状态集合， $P$ 为状态转移函数
  - 马尔可夫决策过程是一个五元组 $<S,P,A,R,\gamma>$， 其中 $R$ 表示从 $S$ 到 $S'$ 能够获得的奖励期望， $\gamma$ 为折扣因子， $A$ 为动作集合
  - 马尔可夫最重要的性质是下一个状态只与当前状态有关，与之前的状态无关，也就是 $p(s{t+1} | s_t)= p(s{t+1}|s_1,s_2,...,s_t)$

### 1.17 如果数据流不具备马尔可夫性质怎么办？应该如何处理？
  - 如果不具备马尔可夫性，即下一个状态与之前的状态也有关，若仅用当前的状态来求解决策过程，势必导致决策的泛化能力变差。为了解决这个问题，可以利用循环神经网络对历史信息建模，获得包含历史信息的状态表征，表征过程也可以使用注意力机制等手段，最后在表征状态空间求解马尔可夫决策过程问题。

### 1.18 最佳价值函数和最佳策略为什么等价呢？
  - 最佳价值函数的定义为 $V^ (s)=\max{\pi} V{\pi}(s)$ ，即我们搜索一种策略 $\pi$ 来让每个状态的价值最大。$V^$ 就是到达每一个状态其的最大价值，同时我们得到的策略就可以说是最佳策略，即 $\pi^{*}(s)=\underset{\pi}{\arg \max }~ V_{\pi}(s)$ 。最佳策略使得每个状态的价值函数都取得最大值。所以如果我们可以得到一个最佳价值函数，就可以说某一个马尔可夫决策过程的环境被解。在这种情况下，其最佳价值函数是一致的，即其达到的上限的值是一致的，但这里可能有多个最佳策略对应于相同的最佳价值

### 1.19 写出基于状态价值函数的贝尔曼方程以及基于动作价值函数的贝尔曼方程

  - 基于状态价值函数的贝尔曼方程: $V{\pi}(s) = \sum{a}{\pi(a|s)}\sum{s',r}{p(s',r|s,a)[r(s,a)+\gamma V{\pi}(s')]}$
  - 基于动作价值函数的贝尔曼方程: $Q{\pi}(s,a)=\sum{s',r}p(s',r|s,a)r(s',a)+\gamma V_{\pi}(s')$

### 1.20 一般怎么求解马尔可夫决策过程？
  - $$V(s)=R(S)+ \gamma \sum_{s' \in S}p(s'|s)V(s')$$. 特别地，其矩阵形式为 $\mathrm{V}=\mathrm{R}+\gamma \mathrm{PV}$。但是贝尔曼方程很难求解且计算复杂度较高，所以可以使用动态规划、蒙特卡洛以及时序差分等方法求解

## 🏆 2. 核心算法  


