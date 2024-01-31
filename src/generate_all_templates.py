from config import *
from audio_capture import *
from plotting import *
from audio_utils import *

def compute_all_template_features():
    all_template_features = {}
    recordings_folder = os.path.join(os.pardir, 'recordings')  
    for digit in range(10):
        for attempt in range(1, 6):  # 对于每个数字的5个模板音频
            filename = f"{digit}-{attempt}.wav"
            file_path = os.path.join(recordings_folder, filename)

            if not os.path.isfile(file_path):
                print(f"File {file_path} does not exist. Skipping.")
                continue

            sample_rate, signal = get_wav_info(file_path)
            features = integrate_mfccs(sample_rate, signal)
            all_template_features[f"{digit}-{attempt}"] = features

    return all_template_features

def save_features(features, directory):
    # 确保目标目录存在
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    for name, feature in features.items():
        feature_path = os.path.join(directory, f"{name}_features.npy")
        np.save(feature_path, feature)
        print(f"Saved features to {feature_path}")

def main():
    all_template_features = compute_all_template_features()
    # 设置保存目录为根目录下的 'features/all_templates'
    save_directory = os.path.join(os.path.dirname(__file__), os.pardir, 'features', 'all_templates')
    save_features(all_template_features, save_directory)

if __name__ == "__main__":
    main()