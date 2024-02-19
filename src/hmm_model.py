# hmm_models.py

import numpy as np
class HMMState:
    def __init__(self, name, emission_probs=None, transition_probs=None):
        self.name = name  # Name or identifier for the state
        self.emission_probs = emission_probs if emission_probs is not None else {}
        self.transition_probs = transition_probs if transition_probs is not None else {}

    def set_emission_probs(self, emissions):
        """
        Set the emission probabilities for this state.
        :param emissions: A dictionary where keys are observations and values are probabilities.
        """
        self.emission_probs = emissions

    def set_transition_probs(self, transitions):
        """
        Set the transition probabilities from this state to other states.
        :param transitions: A dictionary where keys are state names and values are probabilities.
        """
        self.transition_probs = transitions

    def get_emission_prob(self, observation):
        """
        Get the emission probability for a given observation from this state.
        :param observation: The observation for which to get the emission probability.
        :return: The emission probability.
        """
        return self.emission_probs.get(observation, 0)

    def get_transition_prob(self, next_state):
        """
        Get the transition probability to another state.
        :param next_state: The state to transition to.
        :return: The transition probability.
        """
        return self.transition_probs.get(next_state, 0)



class HMM:
    def __init__(self, num_states, feature_dim):
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.state=[]

        # 初始化转移概率矩阵和初始状态概率
        self.A = np.random.dirichlet(np.ones(self.num_states), self.num_states)
        self.pi = np.random.dirichlet(np.ones(self.num_states))

        # 初始化均值和协方差矩阵
        self.means = np.random.randn(self.num_states, self.feature_dim)
        self.covars = np.array([np.eye(self.feature_dim) for _ in range(self.num_states)])

        # 转换为对数概率
        self.log_A = np.log(self.A + 1e-6)
        self.log_pi = np.log(self.pi + 1e-6)
    def add_state(self):
        
    def initialize_parameters(self):
        self.A = np.random.dirichlet(np.ones(self.num_states), self.num_states)
        self.pi = np.random.dirichlet(np.ones(self.num_states))
        self.means = np.random.randn(self.num_states, self.feature_dim)
        for i in range(self.num_states):
            self.covars[i] = np.eye(self.feature_dim) * 1.01


    def update_transition_matrix(self, state_transitions_counts):
        # 使用对数空间更新转移概率矩阵
        total_transitions = state_transitions_counts.sum(axis=1) + 1e-6  # 避免除以零
        self.log_A = np.log(state_transitions_counts + 1e-6) - np.log(total_transitions[:, None])

    def gaussian_probability(self, observation, state):
        mean = self.means[state]
        covar = self.covars[state]
        covar_det = np.linalg.det(covar)
        covar_inv = np.linalg.inv(covar)
        norm_const = 1.0 / (np.power((2 * np.pi), self.feature_dim / 2) * np.sqrt(covar_det))
        prob = np.exp(-0.5 * np.dot(np.dot((observation - mean).T, covar_inv), (observation - mean)))
        return norm_const * prob

    def viterbi(self, observations):
        num_states = self.num_states
        len_observations = len(observations)

        # 动态规划矩阵
        dp_matrix = np.full((num_states, len_observations), float('-inf'))
        path_matrix = np.zeros((num_states, len_observations), dtype=int)

        # 初始化
        for s in range(num_states):
            prob = self.gaussian_probability(observations[0], s)
            if prob <= 0:  # 防止概率为零或非常小
                prob = 1e-10
            dp_matrix[s, 0] = self.log_pi[s] + np.log(prob)

        # 递推
        for t in range(1, len_observations):
            for s in range(num_states):
                max_log_prob = float('-inf')
                best_prev_state = 0
                for prev_state in range(num_states):
                    log_prob = dp_matrix[prev_state, t - 1] + self.log_A[prev_state, s]
                    if log_prob > max_log_prob:
                        max_log_prob = log_prob
                        best_prev_state = prev_state
                prob = self.gaussian_probability(observations[t], s)
                if prob <= 0:  # 同样防止概率为零或非常小
                    prob = 1e-10
                dp_matrix[s, t] = max_log_prob + np.log(prob)
                path_matrix[s, t] = best_prev_state

        # 回溯
        states = np.zeros(len_observations, dtype=int)
        states[-1] = np.argmax(dp_matrix[:, -1])
        for t in range(len_observations - 2, -1, -1):
            states[t] = path_matrix[states[t + 1], t + 1]

        return states
    
    def calculate_score(self, observations):
        num_states = self.num_states
        len_observations = len(observations)

        # 前向概率矩阵
        forward_probs = np.zeros((num_states, len_observations))

        # 初始化
        for s in range(num_states):
            prob = self.gaussian_probability(observations[0], s)
            if prob <= 0:
                prob = 1e-10
            forward_probs[s, 0] = self.pi[s] * prob

        # 递推
        for t in range(1, len_observations):
            for s in range(num_states):
                prob = self.gaussian_probability(observations[t], s)
                if prob <= 0:
                    prob = 1e-10
                forward_probs[s, t] = np.sum(forward_probs[:, t - 1] * self.A[:, s]) * prob

        # 计算最终得分
        score = np.sum(forward_probs[:, -1])
        return score
