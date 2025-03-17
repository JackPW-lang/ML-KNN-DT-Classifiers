import numpy as np
#%matplotlib notebook
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from IPython.core.debugger import set_trace
np.random.seed(1234)

class Node:
    def __init__(self, data_indices, parent):
        self.data_indices = data_indices                    # Stores the data indices in the region defined by this node
        self.left = None                                    # Stores the left child of the node
        self.right = None                                   # Stores the right child of the node
        self.split_feature = None                           # The feature for the split at this node
        self.split_value = None                             # The value of the feature for the split at this node
        if parent:
            self.depth = parent.depth + 1                   # Obtain the depth of the node by adding one to depth of the parent
            self.num_classes = parent.num_classes           # Copies the num_classes from the parent
            self.data = parent.data                         # Copies the data from the parent
            self.labels = parent.labels                     # Copies the labels from the parent
            # Align indices with .loc instead of .iloc
            selected_labels = self.labels.loc[data_indices].values.flatten().astype(int)
            class_prob = np.bincount(selected_labels, minlength=self.num_classes)  # Count frequency of labels in this node's region
            self.class_prob = class_prob / np.sum(class_prob)  # Compute class probabilities



#########################################################################################################
#########################################################################################################
#FEATURE IMPORTANCE CALCULATOR
#########################################################################################################
#########################################################################################################
class FeatureImportanceCalculator:
    def __init__(self, root):
        self.root = root
        self.feature_counts = {}

    def traverse_tree(self, node):
        # Base case: If the node is a leaf, return
        if node is None or (node.left is None and node.right is None):
            return

        # Count the feature used at this node
        if node.split_feature is not None:
            self.feature_counts[node.split_feature] = self.feature_counts.get(node.split_feature, 0) + 1

        # Recursively traverse left and right children
        self.traverse_tree(node.left)
        self.traverse_tree(node.right)

    def compute_importance(self):
        # Start traversal from the root
        self.traverse_tree(self.root)

        # Return sorted features by importance (descending order)
        return sorted(self.feature_counts.items(), key=lambda x: x[1], reverse=True)

#########################################################################################################
#########################################################################################################
#GREEDY NODE SPLIT
#########################################################################################################
#########################################################################################################

def greedy_test(node, cost_fn):
    # Initialize the best parameter values
    best_cost = np.inf
    best_feature, best_value = None, None
    num_instances, num_features = node.data.shape

    # Iterate over features
    for f in range(num_features):
        # Get the data for the f-th feature
        feature_name = node.data.columns[f]
        data_f = node.data.loc[node.data_indices, feature_name]

        # Sort the values for the current feature
        data_sorted = data_f.sort_values()

        # Generate test candidates (averages of consecutive sorted values)
        test_candidates = (data_sorted.values[1:] + data_sorted.values[:-1]) / 2.0

        # Iterate over test candidates
        for test in test_candidates:
            # Split the indices using the test value
            left_indices = data_f.index[data_f <= test]
            right_indices = data_f.index[data_f > test]

            # Skip invalid splits
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue

            # Compute costs for left and right splits
            left_cost = cost_fn(node.labels.loc[left_indices].values)
            right_cost = cost_fn(node.labels.loc[right_indices].values)
            num_left, num_right = len(left_indices), len(right_indices)

            # Calculate weighted combined cost
            cost = (num_left * left_cost + num_right * right_cost) / num_instances

            # Update if a lower cost is found
            if cost < best_cost:
                best_cost = cost
                best_feature = feature_name
                best_value = test

    return best_cost, best_feature, best_value


#########################################################################################################
#########################################################################################################
#COST FUNCTIONS
#########################################################################################################
#########################################################################################################


def cost_misclassification(labels):
    # Ensure labels are a 1D NumPy array of integers
    labels = np.asarray(labels).flatten().astype(int)
    counts = np.bincount(labels)
    class_probs = counts / np.sum(counts)
    return 1 - np.max(class_probs)

def cost_entropy(labels):
    # Ensure labels are a 1D NumPy array of integers
    labels = np.asarray(labels).flatten().astype(int)
    counts = np.bincount(labels)
    class_probs = counts / np.sum(counts)
    class_probs = class_probs[class_probs > 0]  # Remove 0 probabilities to avoid log(0)
    return -np.sum(class_probs * np.log2(class_probs))  # Expression for entropy: -Σ p(x) log2[p(x)]

def cost_gini_index(labels):
    # Ensure labels are a 1D NumPy array of integers
    labels = np.asarray(labels).flatten().astype(int)
    counts = np.bincount(labels)
    class_probs = counts / np.sum(counts)
    return 1 - np.sum(np.square(class_probs))  # Expression for Gini index: 1 - Σ p(x)^2

#########################################################################################################
#########################################################################################################
#CLASSIFIER
#########################################################################################################
#########################################################################################################

class DecisionTree:
    def __init__(self, num_classes=None, max_depth=None, cost_fn=cost_misclassification, min_leaf_instances=1):
        self.max_depth = max_depth      #maximum dept for termination
        self.root = None                #stores the root of the decision tree
        self.cost_fn = cost_fn          #stores the cost function of the decision tree
        self.num_classes = num_classes  #stores the total number of classes
        self.min_leaf_instances = min_leaf_instances  #minimum number of instances in a leaf for termination

    def fit(self, data, labels):
        pass                            #pass in python 3 means nothing happens and the method here is empty

    def predict(self, data_test):
        pass

#########################################################################################################
#########################################################################################################
#FIT
#########################################################################################################
#########################################################################################################

    def fit(self, data, labels, max_depth):
        self.max_depth = max_depth
        self.data = data
        self.labels = labels
        if self.num_classes is None:
            self.num_classes = len(np.unique(labels))
        # Initialize the root of the decision tree
        self.root = Node(data.index, None)  # Use DataFrame index
        self.root.data = data
        self.root.labels = labels
        self.root.num_classes = self.num_classes
        self.root.depth = 0
        # Recursively build the rest of the tree
        self._fit_tree(self.root)
        return self

    def _fit_tree(self, node):
        # Termination condition: leaf node
        if node.depth == self.max_depth or len(node.data_indices) <= self.min_leaf_instances:
            return
        # Greedily select the best test by minimizing the cost
        cost, split_feature, split_value = greedy_test(node, self.cost_fn)
        # If the cost is infinity, terminate
        if np.isinf(cost):
            return
        # Store the split feature and value
        node.split_feature = split_feature
        node.split_value = split_value
        # Get a boolean array indicating which data indices are in the left split
        test = node.data.loc[node.data_indices, split_feature] <= split_value
        # Define left and right child nodes
        left_indices = node.data_indices[test]
        right_indices = node.data_indices[~test]
        left = Node(left_indices, node)
        right = Node(right_indices, node)
        # Recursive call to _fit_tree()
        self._fit_tree(left)
        self._fit_tree(right)
        # Assign the left and right child to the current node
        node.left = left
        node.right = right

#########################################################################################################
#########################################################################################################
#PREDICT
#########################################################################################################
#########################################################################################################

    def predict(self, data_test):
        class_probs = np.zeros((data_test.shape[0], self.num_classes))
        for n, (_, x) in enumerate(data_test.iterrows()):  # Iterate over rows in the DataFrame
            node = self.root
            # Traverse the tree to find the appropriate leaf
            while node.left:
                if x[node.split_feature] <= node.split_value:
                    node = node.left
                else:
                    node = node.right
            # Assign the class probabilities from the leaf node
            class_probs[n, :] = node.class_prob

        return class_probs

    def evaluate_acc(self, probs_test):
        # Make predictions
        y_pred = np.argmax(probs_test, axis=1)

        # Convert y_train and y_test to 1D arrays
        y_train_1d = y_train.squeeze()  # Convert to Series if it's a single-column DataFrame
        y_train_1d = y_train_1d.to_numpy() if isinstance(y_train_1d, pd.Series) else y_train_1d

        y_test_1d = y_test.squeeze()  # Convert to Series if it's a single-column DataFrame
        y_test_1d = y_test_1d.to_numpy() if isinstance(y_test_1d, pd.Series) else y_test_1d

        accuracy = np.sum(y_pred == y_test_1d) / y_test_1d.shape[0]
        print(f'Accuracy is {accuracy * 100:.2f}%')

    def evaluate_auroc(self, y_test, probs_test):
        # Make predictions
        y_pred = np.argmax(probs_test, axis=1)

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)

        # Compute AUROC using the trapezoidal rule
        auroc = 0.0
        for i in range(1, len(fpr)):
            # Trapezoidal rule: Area of each trapezoid
            auroc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2

        print(f"AUROC Score: {auroc:.4f}")


        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUROC = {auroc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="best")
        plt.show()
