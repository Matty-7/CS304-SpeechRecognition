class LexTreeNode:
    def __init__(self, char):
        self.char = char  # 当前节点的字符
        self.children = {}  # 子节点字典，键是字符，值是LexTreeNode
        self.is_end_of_word = False  # 标记这是否是一个单词的结尾

class LexTree:
    def __init__(self):
        self.root = LexTreeNode("*")  # 根节点始终是虚拟字符"*"
    
    def add_word(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = LexTreeNode(char)
            node = node.children[char]
        node.is_end_of_word = True  # 单词结束

    def build_tree(self, words):
        for word in words:
            self.add_word("*" + word)  # 添加单词前加上虚拟字符"*"

# 使用字典构建词汇树的例子
if __name__ == "__main__":
    # 假设dict_words是从文件中读取的字典单词列表
    dict_words = [
        # 这里填充你从dict_1.txt文件中读取的单词
    ]
    lex_tree = LexTree()
    lex_tree.build_tree(dict_words)

    # 你可以在这里添加更多代码来验证树的构建，例如打印树的结构或者添加单词查询功能
