from config import *
from audio_capture import *
from plotting import *
from audio_utils import *

def main():
    # 获取测试特征
    tests = compute_test_features()
    
    # 创建features/tests目录，如果不存在的话
    tests_dir = os.path.join(os.pardir, 'features', 'tests')
    os.makedirs(tests_dir, exist_ok=True)

    # 处理并保存测试特征
    for test_name, features in tests.items():
        # 定义要保存的文件名
        features_filename = f"{test_name}.npy"
        # 定义完整的文件路径
        features_path = os.path.join(tests_dir, features_filename)
        
        # 保存特征向量到.npy文件，便于后续加载和使用
        np.save(features_path, features)
        print(f"Test features for {test_name} saved to {features_path}")

if __name__ == "__main__":
    main()