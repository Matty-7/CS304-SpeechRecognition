import os
from config import *
from audio_capture import *
from plotting import *
from audio_utils import *

# def calculate_accuracy(recognition_results):
#     # 计算正确识别的数量
#     correct_matches = sum([1 for test_name, matched_template in recognition_results.items() if test_name.split('-')[0] in matched_template])
#     # 计算总的测试数量
#     total_tests = len(recognition_results)
#     # 计算准确率
#     accuracy = correct_matches / total_tests
#     return accuracy

def calculate_accuracy(recognition_results):
    correct_matches = 0
    total_tests = 0

    for test_name, matched_template in recognition_results.items():
        if matched_template is None:
            # 跳过因剪枝而没有匹配结果的测试样本
            continue

        # 增加测试计数
        total_tests += 1

        # 检查是否正确匹配
        if test_name.split('-')[0] in matched_template:
            correct_matches += 1

    # 防止除以零的情况
    if total_tests == 0:
        return 0

    return correct_matches / total_tests

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
    prune_thresholds = range(50) # 示例阈值列表，可能需要根据数据调整
    accuracies = []
    
    for prune_threshold in prune_thresholds:
        pruned_results = perform_dtw_recognition_with_pruning(templates, tests, window_size, prune_threshold)
        
        if pruned_results:  # 确保pruned_results不为空
            accuracy = calculate_accuracy(pruned_results)
            accuracies.append(accuracy)
            print(f"Pruning threshold: {prune_threshold}, Accuracy: {accuracy:.2f}")

    # 确保在调用绘图函数前accuracies列表已经填充
    if accuracies:
        plot_pruning_threshold_vs_accuracy(prune_thresholds, accuracies)
    else:
        print("No accuracy data available for plotting.")

    # 执行时间同步DTW剪枝识别并计算准确率
    window_size = 20  # 窗口大小
    prune_thresholds = range(50)
    time_sync_accuracies = []

    for prune_threshold in prune_thresholds:
        time_sync_pruned_results = perform_time_sync_dtw_recognition_with_pruning(templates, tests, window_size, prune_threshold)
        
        if time_sync_pruned_results:  # 确保pruned_results不为空
            time_sync_accuracy = calculate_accuracy(time_sync_pruned_results)
            time_sync_accuracies.append(time_sync_accuracy)
            print(f"Time-Sync Pruning threshold: {prune_threshold}, Accuracy: {time_sync_accuracy:.2f}")

    # 确保在调用绘图函数前accuracies列表已经填充
    if time_sync_accuracies:
        plot_pruning_threshold_vs_accuracy(prune_thresholds, time_sync_accuracies)
    else:
        print("No time-sync accuracy data available for plotting.")

if __name__ == "__main__":
    main()