import os
from config import *
from audio_capture import *
from plotting import *
from audio_utils import *
from dtw_recognition import calculate_accuracy
def main():
    # 加载模板特征
    templates_dir = os.path.join(os.pardir, 'features', 'all_templates')
    templates = load_features(templates_dir)

    # 加载测试特征
    tests_dir = os.path.join(os.pardir, 'features', 'tests')
    tests = load_features(tests_dir)

    # 执行DTW识别
   

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
    prune_threshold = 50 # 示例阈值列表，可能需要根据数据调整
    accuracies = []
    
    
    
    
    

    # 确保在调用绘图函数前accuracies列表已经填充
    

    # 执行时间同步DTW剪枝识别并计算准确率
    window_size = 20  # 窗口大小
   
    time_sync_accuracies = []


    time_sync_pruned_results = perform_time_sync_dtw_recognition_with_pruning(templates, tests, window_size, prune_threshold)
        
    if time_sync_pruned_results:  # 确保pruned_results不为空
        time_sync_accuracy = calculate_accuracy(time_sync_pruned_results)
        time_sync_accuracies.append(time_sync_accuracy)
        print(f"Time-Sync Pruning threshold: {prune_threshold}, Accuracy: {time_sync_accuracy:.2f}")

    # 确保在调用绘图函数前accuracies列表已经填充
    

if __name__ == "__main__":
    main()
#with pruning