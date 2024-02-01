# hmm_training.py

import numpy as np

def train_hmm(hmm, training_data, num_iterations=10, smoothing_constant=1e-6):
    
    initial_state_counts = np.zeros(hmm.num_states)

    for _ in range(num_iterations):
        # 初始化状态转移计数矩阵
        state_transitions_counts = np.zeros((hmm.num_states, hmm.num_states))

        for sample in training_data:
            features = sample['features']
            states = hmm.viterbi(features)

            # 累加状态转移次数
            for t in range(1, len(states)):
                state_transitions_counts[states[t-1], states[t]] += 1

            # 更新发射概率的高斯参数
            for state in range(hmm.num_states):
                # 选择当前状态的观测值
                state_observations = features[states == state]

                # 更新均值和协方差
                if len(state_observations) > 1:
                    hmm.means[state] = np.mean(state_observations, axis=0)
                    hmm.covars[state] = np.cov(state_observations, rowvar=False) + np.eye(hmm.feature_dim) * smoothing_constant

            # 更新初始状态计数
            initial_state_counts[states[0]] += 1

        # 更新转移概率矩阵
        hmm.update_transition_matrix(state_transitions_counts)

        # 根据计数更新初始状态概率，使用平滑处理
        total_counts = initial_state_counts.sum() + (hmm.num_states * smoothing_constant)
        hmm.pi = (initial_state_counts + smoothing_constant) / total_counts
