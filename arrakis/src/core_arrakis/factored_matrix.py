# Have to incorporate it others.
import numpy as np

class FactoredMatrix:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eigenvalues(self):
        # Calculate the eigenvalues of the product of left and right matrices
        return np.linalg.eigvals(np.matmul(self.left, self.right))

    def norm(self, order=2):
        # Calculate the matrix norm of the product of left and right matrices
        return np.linalg.norm(np.matmul(self.left, self.right), ord=order)

    def svd(self):
        # Calculate the SVD of the product of left and right matrices
        return np.linalg.svd(np.matmul(self.left, self.right))