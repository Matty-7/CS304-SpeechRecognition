import os
from config import *
from audio_capture import *
from plotting import *
from audio_utils import *

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

if __name__ == "__main__":
    main()