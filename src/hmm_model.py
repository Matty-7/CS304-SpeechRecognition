# hmm_model.py
import numpy as np

class HMM:
    def __init__(self, num_states, feature_dim=39):
        self.num_states = num_states
        self.feature_dim = feature_dim  # 将 feature_dim 设置为类的属性
        self.A = np.zeros((num_states, num_states))
        self.pi = np.zeros(num_states)
        self.means = np.zeros((num_states, feature_dim))
        self.covars = np.zeros((num_states, feature_dim, feature_dim))
        self.initialize_parameters()

    def initialize_parameters(self):
        # 初始化 A 和 pi
        self.A = np.random.dirichlet(np.ones(self.num_states), self.num_states)
        self.pi = np.random.dirichlet(np.ones(self.num_states))
        # 初始化均值和协方差矩阵
        self.means = np.random.randn(self.num_states, self.feature_dim)
        for i in range(self.num_states):
            self.covars[i] = np.eye(self.feature_dim)

    def update_transition_matrix(self, state_transitions):
        # state_transitions 是一个矩阵，表示状态i转移到状态j的次数
        total_transitions = state_transitions.sum(axis=1)
        self.A = state_transitions / total_transitions[:, None]

    def update_emission_matrix(self, observations, state_assignments):
        # 这里我们简化处理，假设每个状态的观测值是离散的
        # 对于连续值或更复杂的情况，需要使用不同的方法
        for i in range(self.num_states):
            state_obs = observations[state_assignments == i]
            if len(state_obs) > 0:
                self.B[i, :] = np.bincount(state_obs, minlength=self.num_observations) / len(state_obs)

    def update_initial_state_distribution(self, initial_state_counts):
        total_counts = initial_state_counts.sum()
        self.pi = initial_state_counts / total_counts

    # Viterbi 算法来确定状态序列，以及计算观测序列概率等
    def viterbi(hmm, observations):
        num_states = hmm.num_states
        len_observations = len(observations)

        # dp_matrix 存储每个状态的最大概率
        dp_matrix = np.zeros((num_states, len_observations))

        # path_matrix 用于回溯最优路径
        path_matrix = np.zeros((num_states, len_observations), dtype=int)

        # 初始化
        dp_matrix[:, 0] = hmm.pi * hmm.B[:, observations[0]]

        # 递推
        for t in range(1, len_observations):
            for s in range(num_states):
                prob = dp_matrix[:, t - 1] * hmm.A[:, s] * hmm.B[s, observations[t]]
                dp_matrix[s, t] = np.max(prob)
                path_matrix[s, t] = np.argmax(prob)

        # 回溯
        states = np.zeros(len_observations, dtype=int)
        states[-1] = np.argmax(dp_matrix[:, -1])
        for t in range(len_observations - 2, -1, -1):
            states[t] = path_matrix[states[t + 1], t + 1]

        return states

# 如何从观测数据中提取状态转移和状态分配的函数，以及如何计算模型的前向和后向概率等
