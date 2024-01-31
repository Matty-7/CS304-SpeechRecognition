from config import *
from audio_capture import *
from plotting import *
from audio_utils import *

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