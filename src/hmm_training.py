# hmm_training.py

import numpy as np

def train_hmm(hmm, training_data, num_iterations=10, smoothing_constant=1e-6):
    # 初始化初始状态计数为零
    initial_state_counts = np.zeros(hmm.num_states)

    for _ in range(num_iterations):
        for sample in training_data:
            # 使用 Viterbi 算法获取最可能的状态序列
            states = hmm.viterbi(sample)
            
            # 更新转移概率矩阵
            hmm.update_transition_matrix(states)

            # 更新发射概率的高斯参数
            for state in range(hmm.num_states):
                # 选择当前状态的观测值
                state_observations = sample[states == state]

                # 更新均值和协方差（如果有足够的观测值）
                if len(state_observations) > 1:
                    hmm.means[state] = np.mean(state_observations, axis=0)
                    hmm.covars[state] = np.cov(state_observations, rowvar=False)

            # 更新初始状态计数
            initial_state_counts[states[0]] += 1

    # 根据计数更新初始状态概率，使用平滑处理
    total_counts = initial_state_counts.sum() + (hmm.num_states * smoothing_constant)
    hmm.pi = (initial_state_counts + smoothing_constant) / total_counts