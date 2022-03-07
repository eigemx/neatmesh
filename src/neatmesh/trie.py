from typing import List

class TrieNode:
    __slots__ = ['index', 'is_end', 'children']

    def __init__(self, index: int):
        # the character stored in this node
        self.index = index

        # whether this can be the end of a word
        self.is_end = False

        self.children = {}


class Trie:
    __slots__ = ['root']

    def __init__(self):
        self.root = TrieNode(-1)
    
    def insert(self, face: List[int]):
        """Insert a word into the trie"""
        node = self.root
        
        # Loop through each point in the face
        # Check if there is no child containing the point, 
        # create a new child for the current node
        for point in face:
            if point in node.children:
                node = node.children[point]
            else:
                # If a character is not found,
                # create a new node in the trie
                new_node = TrieNode(point)
                node.children[point] = new_node
                node = new_node
        
        # Mark the end of a word
        node.is_end = True

        
    def search(self, face: List[int]) -> bool:
        node =self.root
        
        for point in face:
            if point in node.children:
                node = node.children[point]
                continue
            else:
                return False
        
        if node.is_end:
            return True
        
        return False

if __name__ == '__main__':
    trie = Trie()
    trie.insert([1, 3, 4])
    trie.insert([1, 7, 8])
    print(trie.search([1,3]))
