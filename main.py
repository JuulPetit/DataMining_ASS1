class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, gain=0):
        self.feature = feature      # Index of the feature to split on
        self.threshold = threshold  # Threshold value to split the feature (welke waarde)
        self.left = left            # Left child (Node)
        self.right = right          # Right child (Node)
        self.value = value          # Predicted class if this is a leaf node (1 0f 0)
        self.gain = gain

class Tree:
    def __init__():
        pass

    
if __name__ == "__main__":
    pass
