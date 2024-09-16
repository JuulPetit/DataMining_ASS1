from TreeNode import TreeNode
import numpy as np

class DecisionTree():
    def __init__(self, nmin, minleaf, nfeat) -> None:
        """
        Class that implements the tree-growing algorithm. 

        Args:  
            nmin: The number of observations that a node must contain at least, for it to be allowed to be split.
            minleaf: The minimum number of observations required for a leaf node.
            nfeat: The number of features that should be considered for each split.
        """
        self.nmin = nmin
        self.minleaf = minleaf
        self.nfeat = nfeat
        self.root = None

    def fit(self, x, y):
        """
        Fit the tree to the data (start growing the tree).
        
        Args: 
            x: Data Matrix (features).
            y: Class labels (binary).
        """
        print("Dataset:", x)
        print("Dataset:", y)

        # Call the function to recursively grow the tree
        self.root = self._grow_tree(x, y)
    
    def _gini(self, y):
        """
        Compute the Gini index for the given labels (y). 
        
        Args: 
            y: Array of binary class labels.
        
        Returns: 
            Gini index value. 
        """
        n = len(y)
        if n == 0:
            return 0
        # Assign probabilities
        p1 = np.sum(y == 1) / n
        p0 = 1 - p1
        return 1 - (p1 ** 2 + p0 ** 2)

    def _best_split(self, x, y):
        """
        Find the best feature and threshold to split on.
        
        Args:
            x: Data Matrix (Features).
            y: Class Labels (Binary).
        
        Returns:
            best_feature: Index of the best feature to split on.
            best_threshold: Threshold value for the split.
            best_gain: The Gini gain from the split. 
        """
        m, n = x.shape

        # Check if the node has fewer observations than mnin, return no split
        if m < self.nmin: 
            return None, None, 0
        
        best_gain = 0
        best_feature = None
        best_threshold = None
        current_gini = self._gini(y)

        # Randomly select nfeat features to consider for the split
        features = np.random.choice(n, self.nfeat, replace=False)

        # Find the best split among the selected features
        for feature in features:
            thresholds = np.unique(x[:, feature]) # Unique threshold values from features

            # Loop over each threshold value to evaluate the split
            for threshold in thresholds:
                # Split the data based on the current threshold
                left_split = x[:, feature] < threshold
                right_split = ~left_split

                # Check both children nodes have at least minleaf observations
                if np.sum(left_split) < self.minleaf or np.sum(right_split) < self.minleaf:
                    continue # Don't allow splits that don't satisfy minleaf

                # Calculate Gini for the left and right splits
                gini_left = self._gini(y[left_split])
                gini_right = self._gini(y[right_split])

                # Weighted Gini for the split
                weighted_gini = (np.sum(left_split) * gini_left + np.sum(right_split) * gini_right) / m

                # Calculate Gini gain
                gain = current_gini - weighted_gini

                # Update best split if better one is found
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_gain, best_feature, best_threshold

    def _grow_tree(self, x, y):
        """
        Grow a classification tree by recursively splitting the nodes. 

        Args: 
            x: Data Matrix (Features).
            y: Class labels (binary).

        Returns:
            A TreeNode that represents the root of the tree.
        """
        # Print de huidige dataset en labels voor elke node
        print(f"Current node dataset (size: {len(y)}):\n{x}")
        print(f"Current node labels:\n{y}")

        # Check stopping criteria (nmin, minleaf)
        if len(y) < self.nmin or np.all( y == y[0]):
            # Create a lead node if number of observations is less than nmin, or if all class labels are the same
            majority_class = np.argmax(np.bincount(y))
            return TreeNode(gini=self._gini(y), c_label=majority_class)

        # Find the best split on nfeat features
        gain, split_feature, split_value = self._best_split(x, y)

        # Print the split details
        print(f"Best split: feature {split_feature} with threshold {split_value} (gain: {gain})")
        
        # If no split is possible, return a leaf node
        if gain == 0: 
            majority_class = np.argmax(np.bincount(y))
            return TreeNode(gini=self._gini(y), c_label=majority_class)

        # Otherwise, create the left and right children recursively based on best feature and threshold
        left_split = x[:, split_feature] < split_feature
        right_split = ~left_split
        
        # Check whether left and right split contain data
        print(f"Left split size: {np.sum(left_split)}, Right split size: {np.sum(right_split)}")
        
        # Recursively continue growing left and right branch
        if np.sum(left_split) == 0 or np.sum(right_split) == 0:
            # Create a leaf node
            majority_class = np.argmax(np.bincount(y))
            print(f"One split is empty, creating leaf node with class: {majority_class}")
            return TreeNode(gini=self._gini(y), c_label=majority_class)
        
        left_node = self._grow_tree(x[left_split], y[left_split])
        right_node = self._grow_tree(x[right_split], y[right_split])

        # Return the node with split information
        print(f"Node with split on feature {split_feature} and threshold {split_value}")
        node = TreeNode(gini=self._gini(y), split_feature=split_feature, split_value=split_value)
        node.left = left_node
        node.right = right_node
        return node
