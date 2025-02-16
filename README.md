# 🌞 求职强化学习算法工程师  CPNT睡觉王

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

### 1.4 部分可观测马尔可夫决策过程（partially observable Markov decision process，POMDP）?
  - 即马尔可夫决策过程的泛化。部分可观测马尔可夫决策过程依然具有马尔可夫性质，但是其假设智能体无法感知环境的状态，只能知道部分观测值

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

## 🏆 2. 核心算法  


