import numpy as np
from hmmlearn import hmm

n_components = 5


models = {}

for digit in range(10):  
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")
    X = np.empty((0, 39))  
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


total_tests = 50  # 总测试样本数
accuracy = accuracy / total_tests
print(f"Recognition Accuracy: {accuracy * 100}%")
