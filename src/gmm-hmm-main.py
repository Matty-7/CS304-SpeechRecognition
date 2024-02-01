import numpy as np
from hmmlearn import hmm

# 假设n_components是HMM的状态数，n_mix是每个状态的高斯混合数
n_components = 5
n_mix = 4  # 每个状态4个高斯混合

# 初始化模型存储字典
models = {}

for digit in range(10):  # 对于每个数字
    # 使用GMMHMM，并设置n_mix，调整covars_prior以增加正则化
    model = hmm.GMMHMM(n_components=n_components, 
                       n_mix=n_mix, 
                       covariance_type="diag", 
                       )
    X = np.empty((0, 39))  # 假设特征向量的维度是39
    lengths = []  # 存储每个序列的长度
    for i in range(1, 6):  # 读取每个数字的5个训练样本
        features = np.load(f"../features/all_templates/{digit}-{i}.npy")
        X = np.vstack((X, features))  # 堆叠特征
        lengths.append(len(features))  # 添加序列长度
    model.fit(X, lengths)  # 使用正确的长度
    models[digit] = model

# 测试模型
accuracy = 0
for digit in range(10):  # 对于每个数字的测试数据
    for i in range(6, 11):  # 测试样本
        filename = f"../features/tests/{digit}-{i}.npy"
        test_feature = np.load(filename)
        log_probs = [model.score(test_feature) for model in models.values()]
        pred_digit = np.argmax(log_probs)
        if pred_digit == digit:
            accuracy += 1

# 计算准确率
total_tests = 50  # 总测试样本数
accuracy = accuracy / total_tests
print(f"Recognition Accuracy: {accuracy * 100}%")
