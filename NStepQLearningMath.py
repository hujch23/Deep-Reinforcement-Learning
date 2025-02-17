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