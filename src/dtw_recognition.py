import os
from config import *
from audio_capture import *
from plotting import *
from audio_utils import *

def calculate_accuracy(recognition_results):
    # 计算正确识别的数量
    correct_matches = sum([1 for test_name, matched_template in recognition_results.items() if test_name.split('-')[0] in matched_template])
    # 计算总的测试数量
    total_tests = len(recognition_results)
    # 计算准确率
    accuracy = correct_matches / total_tests
    return accuracy

def main():
    # 加载模板特征
    templates_dir = os.path.join(os.pardir, 'features', 'templates')
    templates = load_features(templates_dir)

    # 加载测试特征
    tests_dir = os.path.join(os.pardir, 'features', 'tests')
    tests = load_features(tests_dir)

    # 执行DTW识别
    recognition_results = perform_dtw_recognition(templates, tests)

    # 打印识别结果
    for test_name, matched_template in recognition_results.items():
        print(f"Test {test_name} is recognized as {matched_template}")

    # 计算并打印识别正确率
    accuracy = calculate_accuracy(recognition_results)
    print(f"Recognition accuracy: {accuracy:.2f}")

    # 执行时间同步DTW识别
    window_size = 10  # 窗口大小可能需要根据你的序列长度调整
    time_sync_results = perform_time_sync_dtw_recognition(templates, tests, window_size)

    # 打印时间同步DTW的识别结果
    for test_name, matched_template in time_sync_results.items():
        print(f"Time-sync test {test_name} is recognized as {matched_template}")

    # 计算并打印时间同步DTW的识别正确率
    time_sync_accuracy = calculate_accuracy(time_sync_results)
    print(f"Time-sync DTW recognition accuracy: {time_sync_accuracy:.2f}")

    # 确定窗口大小和剪枝阈值
    window_size = 10  # 窗口大小可能需要根据你的序列长度调整
    prune_thresholds = [np.inf, 100, 50, 25, 10, 5]  # 示例阈值列表，可能需要根据数据调整
    
    # 对每个剪枝阈值执行DTW识别并计算准确率
    for prune_threshold in prune_thresholds:
        pruned_results = perform_dtw_recognition_with_pruning(templates, tests, window_size, prune_threshold)
        
        # 计算并打印剪枝DTW的识别正确率
        pruned_accuracy = calculate_accuracy(pruned_results)
        print(f"Pruning threshold: {prune_threshold}, Accuracy: {pruned_accuracy:.2f}")

if __name__ == "__main__":
    main()