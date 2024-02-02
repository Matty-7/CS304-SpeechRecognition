class LexTreeNode:
    def __init__(self, char):
        self.char = char  # 当前节点的字符
        self.children = {}  # 子节点字典，键是字符，值是LexTreeNode
        self.is_end_of_word = False  

class LexTree:
    def __init__(self):
        self.root = LexTreeNode("*")  # 根节点始终是虚拟字符"*"
    
    def add_word(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = LexTreeNode(char)
            node = node.children[char]
        node.is_end_of_word = True 

    def build_tree(self, words):
        for word in words:
            self.add_word("*" + word)

    def search_word(self, word):
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                return False
            
        return node.is_end_of_word
    
    def check_spelling(self, word):
        if self.search_word("*" + word):
            return True, "Word is spelled correctly."
        else:
            return False, "Word might be spelled incorrectly."
        
    def find_suggestions(self, word, beam_width=3):
        candidates = [(self.root, "*", 0)]  # 初始化候选列表（节点，累积单词，评分为0）
        suggestions = []

        for char in '*' + word:  # 包含虚拟字符'*'
            new_candidates = []
            for node, acc_word, _ in candidates:
                for child_char, child_node in node.children.items():
                    new_word = acc_word + child_char
                    # 计算编辑距离作为新的评分
                    score = levenshtein_distance(new_word, '*' + word)
                    new_candidates.append((child_node, new_word, score))

            # 按编辑距离排序并保留评分最低的beam_width个候选词
            candidates = sorted(new_candidates, key=lambda x: x[2])[:beam_width]

        # 收集所有完整的单词建议
        for node, acc_word, score in candidates:
            if node.is_end_of_word:
                suggestions.append((acc_word[1:], score))  # 移除虚拟字符'*'

        # 按评分排序并返回最佳建议
        suggestions = sorted(suggestions, key=lambda x: x[1])
        best_suggestions = [suggestion for suggestion, _ in suggestions]

        return best_suggestions[:beam_width]

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

if __name__ == "__main__":
    dict_file_path = '../lextree/dict_1.txt'
    dict_words = []

    # 从文件加载字典单词到词汇树
    with open(dict_file_path, 'r', encoding='latin1') as file:
        for line in file:
            word = line.strip()
            if word:  # 确保单词不为空
                dict_words.append(word)

    lex_tree = LexTree()
    lex_tree.build_tree(dict_words)

    # 测试单词列表，包括正确拼写和错误拼写的单词
    test_words = ["able", "ablle", "ble", "abolishing", "abbolishing"]
    
    for word in test_words:
        # 首先检查单词拼写是否正确
        correct, message = lex_tree.check_spelling(word)
        print(f"'{word}': {message}")
        
        # 如果拼写错误，尝试找到建议
        if not correct:
            suggestions = lex_tree.find_suggestions(word)
            if suggestions:
                print(f"Did you mean: {', '.join(suggestions)}?")
            else:
                print("No suggestions available.")


