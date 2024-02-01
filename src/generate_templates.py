# generate_templates.py

from config import *
from audio_capture import *
from plotting import *
from audio_utils import *

def compute_template_features():
    template_features = {}
    # 设置 recordings 文件夹相对于当前文件的路径
    recordings_folder = os.path.join(os.pardir, 'recordings')  
    for digit in range(10):
        filename = f"{digit}-1.wav"
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
        template_features[digit] = features
    
    return template_features

def main():
    # 获取模板特征
    templates = compute_template_features()
    
    # 创建features目录及其子目录
    features_dir = os.path.join(os.pardir, 'features')
    templates_dir = os.path.join(features_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)  # 创建templates目录

    # 处理并保存模板特征
    for digit, features in templates.items():
        # 定义要保存的文件名
        features_filename = f'template_features_digit_{digit}.npy'
        # 定义完整的文件路径
        features_path = os.path.join(templates_dir, features_filename)
        
        # 保存特征向量到.npy文件，便于后续加载和使用
        np.save(features_path, features)
        print(f'Template features for digit {digit} saved to {features_path}')


if __name__ == "__main__":
    main()
