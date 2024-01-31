# hmm_main.py

from hmm_model import *
from hmm_training import *
from hmm_recognition import *
from audio_utils import *

def hmm_load_features(data_dir):
    # 创建一个空列表来存储样本对象
    samples = []

    # 循环读取数据并创建样本对象，每个样本对象包括特征数据和标签
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.npy'):
            # 解析文件名，获取标签信息
            parts = file_name.split('_')
            if len(parts) == 3 and parts[2].endswith('.npy'):
                label = int(parts[0])
                
                # 加载特征数据
                features = np.load(os.path.join(data_dir, file_name))

                # 创建样本对象并添加到列表中
                sample = {'label': label, 'features': features}
                samples.append(sample)

    return samples

def main():

    # 定义数字类别和状态数
    num_digits = 10
    num_states = 5
    num_observations = 10  # 或者特征向量的长度

    # 加载训练和测试数据
    training_data = hmm_load_features('../features/all_templates')
    test_data = hmm_load_features('../features/tests')

    # 初始化HMM实例列表，每个数字一个HMM模型
    hmms = [HMM(num_states, num_observations) for _ in range(num_digits)]

    # 训练HMM模型，每个数字一个模型
    for digit in range(num_digits):
        digit_data = [sample for sample in training_data if sample['label'] == digit]
        train_hmm(hmms[digit], digit_data)

    # 使用HMM进行识别并计算准确率
    num_correct = 0
    total_predictions = 0

    if len(test_data) == 0:
        print("No test data available. Cannot calculate accuracy.")
    else:
        for test_sample in test_data:
            recognized_digit = None
            max_score = float('-inf')

            for digit in range(num_digits):
                score = hmms[digit].calculate_score(test_sample['features'])

                if score > max_score:
                    max_score = score
                    recognized_digit = digit

            total_predictions += 1
            if recognized_digit == test_sample['label']:
                num_correct += 1

        accuracy = (num_correct / total_predictions) * 100

        print(f"Recognition accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()