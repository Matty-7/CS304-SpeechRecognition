from config import *
from audio_capture import *
from plotting import *
from audio_utils import *

def compute_test_features():
    test_features = {}
    recordings_folder = os.path.join(os.pardir, 'recordings')  
    for digit in range(10):
        for attempt in range(6, 11):  # 对于每个数字的5个测试文件
            filename = f"{digit}-{attempt}.wav"
            file_path = os.path.join(recordings_folder, filename)

            # 确保文件存在
            if not os.path.isfile(file_path):
                print(f"File {file_path} does not exist. Skipping.")
                continue

            # 加载音频文件
            sample_rate, signal = get_wav_info(file_path)
            
            # 计算特征向量序列
            features = integrate_mfccs(sample_rate, signal)
            
            # 保存特征向量序列
            test_features[f"{digit}-{attempt}"] = features

    return test_features

def main():
    
    tests = compute_test_features()
    
    tests_dir = os.path.join(os.pardir, 'features', 'tests')
    os.makedirs(tests_dir, exist_ok=True)

    # 处理并保存测试特征
    for test_name, features in tests.items():
        # 定义要保存的文件名
        features_filename = f"{test_name}.npy"
        # 定义完整的文件路径
        features_path = os.path.join(tests_dir, features_filename)
        
        # 保存特征向量到.npy文件
        np.save(features_path, features)
        print(f"Test features for {test_name} saved to {features_path}")

if __name__ == "__main__":
    main()