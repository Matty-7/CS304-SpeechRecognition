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
        candidates = [(self.root, "", 0)]  # 初始化候选列表（节点，累积单词，评分）
        suggestions = []  # 最终建议列表

        # 对每个字符进行扩展搜索
        for char in word:
            new_candidates = []
            for node, acc_word, score in candidates:
                # 检查当前节点的每个子节点
                for child_char, child_node in node.children.items():
                    new_score = score + (1 if child_char == char else 0)
                    new_candidates.append((child_node, acc_word + child_char, new_score))
            
            # 保留评分最高的beam_width个候选词
            candidates = sorted(new_candidates, key=lambda x: x[2], reverse=True)[:beam_width]
        
        # 在候选词中找到完整的单词
        for node, acc_word, score in candidates:
            if node.is_end_of_word:
                suggestions.append((acc_word, score))
        
        # 按评分排序并返回最佳建议
        suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
        best_suggestions = [suggestion for suggestion, score in suggestions]
        
        return best_suggestions[:beam_width]  # 返回评分最高的几个建议


# 使用字典构建词汇树的例子
if __name__ == "__main__":
    dict_file_path = '../lextree/dict_1.txt'
    dict_words = []

    with open(dict_file_path, 'r', encoding='latin1') as file:
        for line in file:
            word = line.strip()
            if word:
                dict_words.append(word)

    lex_tree = LexTree()
    lex_tree.build_tree(dict_words)

    test_words = ["able", "ablle", "abolishing", "abbolishing", "orange"]
    for word in test_words:
        correct, message = lex_tree.check_spelling(word)
        print(f"'{word}': {message}")

    # 例如打印树的结构或者添加单词查询功能
