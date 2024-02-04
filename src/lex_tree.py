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
            self.add_word(word)  # 移除"*"，直接添加单词

    def print_tree(self, node=None, indent="", is_last=True, is_root=True):
        if not node:
            node = self.root
        
        # 如果不是根节点，则打印当前节点的字符
        if not is_root:
            prefix = '    ' if is_last else '|   '
            print(indent[:-4] + prefix + '--' + node.char)
            indent += '    ' if is_last else '|   '
        
        # 如果当前节点是单词的结尾，并且有子节点，打印一个竖线
        if node.is_end_of_word and node.children:
            print(indent[:-4] + '|')
        
        # 递归打印子节点，除了最后一个外，所有的子节点后面都会加'|'
        child_count = len(node.children)
        for i, (child_char, child_node) in enumerate(node.children.items(), 1):
            self.print_tree(child_node, indent, i == child_count, False)

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
    
    def segment_and_spellcheck(self, text, beam_width=3):
        candidates = [(self.root, "", 0, [])]  # (node, accumulated word, score, list of words)
        for char in text:
            new_candidates = []
            for node, acc_word, score, words in candidates:
                node_matched = False
                for child_char, child_node in node.children.items():
                    if child_char == char:
                        new_candidates.append((child_node, acc_word + char, score, words))
                        node_matched = True
                    elif child_char == '*' and node.is_end_of_word:
                        # Transition to start a new word, if the current node marks end of a word
                        new_candidates.append((self.root, char, score, words + [acc_word]))
                        node_matched = True
                if not node_matched:
                    # Penalize unmatched characters
                    new_candidates.append((node, acc_word + char, score + 1, words))
            
            # Prune to keep top beam_width candidates
            candidates = sorted(new_candidates, key=lambda x: x[2])[:beam_width]
        
        # Attempt to finalize words if the last node was end-of-word
        final_candidates = [(node, acc_word, score, words + [acc_word]) for node, acc_word, score, words in candidates if node.is_end_of_word or acc_word == ""]

        if not final_candidates:  # Fallback if no ideal candidates found
            final_candidates = candidates  # This might need more sophisticated handling

        # Choose the best candidate based on the score; fallback to any candidate if empty
        best_candidate = sorted(final_candidates, key=lambda x: x[2])[0] if final_candidates else candidates[0]
        _, _, _, best_words = best_candidate

        return ' '.join(best_words).strip()


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

def load_dictionary(file_path):
    """
    从给定路径加载字典文件。
    :param file_path: 字典文件的路径。
    :return: 包含所有字典单词的列表。
    """
    dict_words = []
    with open(file_path, 'r', encoding='latin1') as file:  # 使用'latin1'编码以确保兼容性
        for line in file:
            word = line.strip()  # 移除每行末尾的换行符
            if word:  # 确保单词不为空
                dict_words.append(word)
    return dict_words

def load_text(file_path):
    """
    从给定路径加载文本文件。
    :param file_path: 文本文件的路径。
    :return: 文本内容的字符串。
    """
    with open(file_path, 'r', encoding='latin1') as file:
        return file.read().strip()  # 读取整个文件内容，并移除首尾的空白字符

if __name__ == "__main__":
    # 加载字典
    dict_words = load_dictionary('../lextree/dict_1.txt')
    dict_words_5 = dict_words[:5]
    words = ["a", "an", "and", "apple", "bat", "battle", "banana"]
    lex_tree = LexTree()
    lex_tree.build_tree(words)

    print("Printing the Lexical Tree structure...")
    lex_tree.print_tree()

    # 文件路径
    unsegmented_files = ['../lextree/unsegmented0.txt', '../lextree/unsegmented.txt']
    beam_widths = [5, 10, 15]

    # 对每个文件和每个 beam_width 运行实验
    for file_path in unsegmented_files:
        text = load_text(file_path)
        for beam_width in beam_widths:
            segmented_text = lex_tree.segment_and_spellcheck(text, beam_width)
            print(f"Results for {file_path} with beam_width {beam_width}:")
            print(segmented_text)
            # 评估结果并与正确的分割进行比较
            # 实现比较和评估的代码部分根据具体需求自定义

