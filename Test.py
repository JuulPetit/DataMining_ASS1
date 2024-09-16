import numpy as np
from DecisionTree import DecisionTree

# Voorbeeld dataset
data = np.array([
    [22, 0, 0, 28, 1],  # Features: age, married, house, income, gender
    [46, 0, 1, 32, 0],
    [24, 1, 1, 24, 1],
    [25, 0, 0, 27, 1],
    [29, 1, 1, 32, 0],
    [45, 1, 1, 30, 0],
    [63, 1, 1, 58, 1],
    [36, 1, 0, 52, 1],
    [23, 0, 1, 40, 0],
    [50, 1, 1, 28, 0]
])

# Labels (class)
labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Instantieer DecisionTree met nmin, minleaf, en nfeat
tree = DecisionTree(nmin=2, minleaf=1, nfeat=3)

# Fit de decision tree op de data
tree.fit(data, labels)

# Je zou nu de tree structuur kunnen visualiseren of traverseren om te zien of de boom goed is gegroeid
def traverse_tree(node, depth=0):
    """Recursively traverse the tree to print its structure."""
    if node.is_terminal():
        print(f"{'  '*depth}Leaf node with class: {node.c_label}, Gini: {node.gini}")
    else:
        print(f"{'  '*depth}Split on feature {node.split_feature} with threshold {node.split_value}, Gini: {node.gini}")
        traverse_tree(node.left, depth + 1)
        traverse_tree(node.right, depth + 1)

# Visualiseer de boom
traverse_tree(tree.root)
