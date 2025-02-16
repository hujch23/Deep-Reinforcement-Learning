# 🌞 求职强化学习算法工程师  CPNT睡觉王

## 1. 基础概念  

1.1 强化学习的基本组成部分有哪些？:主要包含以下组件：  

Agent（智能体）  Environment（环境）   State（状态）  Action（动作）   Reward（奖励）   Policy（策略）  


1.2 解释什么是回报（Return）和价值函数（Value Function）？  
  - 回报：从某一时刻开始，未来所有奖励的折扣总和  
  - 价值函数：从某个状态开始，遵循特定策略能够获得的期望回报  

1.3 全部可观测（full observability）、完全可观测（fully observed）和部分可观测（partially observed）?
  - 当智能体的状态与环境的状态等价时，我们就称这个环境是全部可观测的
  - 当智能体能够观察到环境的所有状态时，我们称这个环境是完全可观测的
  - 一般智能体不能观察到环境的所有状态时，我们称这个环境是部分可观测的


- **Q:** 部分可观测马尔可夫决策过程（partially observable Markov decision process，POMDP）?
- **A:** 即马尔可夫决策过程的泛化。部分可观测马尔可夫决策过程依然具有马尔可夫性质，但是其假设智能体无法感知环境的状态，只能知道部分观测值。





## 2. 经典算法  

### 2.1 价值迭代类  
- **Q:** 介绍Q-learning算法的原理和特点  
- **A:**  
  - 基于值迭代的离线策略（off-policy）算法  
  - 使用Q表格存储状态-动作对的价值  
  - 通过TD更新来学习最优动作价值函数  
  - 更新公式：`Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]`  

### 2.2 策略梯度类  
- **Q:** REINFORCE算法的原理是什么？  
- **A:**  
  - 基于蒙特卡洛采样的策略梯度算法  
  - 直接优化策略函数  
  - 使用整个回合的累积奖励作为梯度估计  
  - 高方差是其主要缺点  

### 2.3 Actor-Critic方法  
- **Q:** Actor-Critic框架的优势是什么？  
- **A:**  
  - 结合了策略梯度和值函数逼近  
  - Actor负责策略更新，Critic负责值函数评估  
  - 相比纯策略梯度方法具有更低的方差  
  - 相比纯值函数方法可以处理连续动作空间  

## 3. 深度强化学习  

### 3.1 DQN及其变体  
- **Q:** DQN的主要创新点是什么？  
- **A:**  
  - 使用深度神经网络逼近Q函数  
  - 经验回放机制（Experience Replay）  
  - 目标网络（Target Network）  
  - 解决了深度学习与Q-learning结合的不稳定性问题  

### 3.2 高级算法  
- **Q:** 比较PPO和TRPO的异同  
- **A:**  
  - 都属于信任域策略优化方法  
  - PPO使用裁剪目标函数，实现更简单  
  - TRPO使用KL散度约束，理论性更强  
  - PPO通常是首选，因为实现简单且效果好  

## 4. 实践问题  

### 4.1 调优技巧  
- **Q:** 强化学习算法调参的关键点有哪些？  
- **A:**  
  - 学习率的选择和调整  
  - 奖励函数的设计  
  - 探索策略的选择（ε-greedy, 玻尔兹曼探索等）  
  - 神经网络架构的设计  
  - 批量大小的选择  

### 4.2 常见问题  
- **Q:** 如何处理奖励稀疏问题？  
- **A:**  
  - 奖励塑形（Reward Shaping）  
  - 分层强化学习  
  - 好奇心驱动的探索  
  - 逆向强化学习  
  - 模仿学习  

### 4.3 评估指标  
- **Q:** 如何评估强化学习算法的性能？  
- **A:**  
  - 累积奖励/平均奖励  
  - 收敛速度  
  - 样本效率  
  - 稳定性  
  - 泛化能力  

## 5. 前沿发展  

### 5.1 多智能体强化学习  
- **Q:** 多智能体强化学习面临的主要挑战是什么？  
- **A:**  
  - 非平稳环境  
  - 信用分配问题  
  - 通信协议学习  
  - 策略协调  

### 5.2 模型基强化学习  
- **Q:** 模型基强化学习的优势是什么？  
- **A:**  
  - 更高的样本效率  
  - 可以进行规划  
  - 更好的泛化性  
  - 可解释性更强  
