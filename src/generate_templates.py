# generate_templates.py

from config import *
from audio_capture import *
from plotting import *
from audio_utils import *

def main():
    # 获取模板特征
    templates = compute_template_features()

    # 处理并保存/返回模板特征
    # 这里你可以选择打印特征、保存到文件或者其他处理
    for digit, features in templates.items():
        print(f"Template features for digit {digit}:")
        print(features)
        # 保存特征到文件的代码可以放在这里

if __name__ == "__main__":
    main()
