import numpy as np
import matplotlib.pyplot as plt

class WSPFCM:
    def __init__(self, n_clusters=3, lambda1=1.0, lambda2=1.0, p=2.0, q=2.0,
                 max_iter=100, epsilon=1e-5, random_state=None):
        self.n_clusters = n_clusters
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.p = p  # fuzziness degree
        self.q = q  # possibilistic degree
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_state = random_state

    def fit(self, X, labels=None):
        """
        X: array-like, shape (n_samples, n_features)
        labels: array-like of length n_samples. For labeled samples, provide the true class (assumed to be in {0,...,n_clusters-1});
                for unlabeled samples, set label to None.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n, d = X.shape

        # Construct b (indicator for labeled data) and f (fuzzy label matrix for supervised term)
        b = np.zeros(n)
        f = np.zeros((n, self.n_clusters))
        if labels is not None:
            for j in range(n):
                if labels[j] is not None:
                    b[j] = 1
                    # Here we assume the provided label is an integer in 0 ... n_clusters-1.
                    f[j, int(labels[j])] = 1

        # Initialize cluster centers by randomly selecting data points
        init_indices = np.random.choice(n, self.n_clusters, replace=False)
        centers = X[init_indices].copy()

        # Initialize membership (m) and typicality (t) matrices.
        m = np.full((n, self.n_clusters), 1.0 / self.n_clusters)
        t = np.full((n, self.n_clusters), 1.0)

        # Precompute global statistics: mean of X and sum of squared distances to mean.
        x_mean = np.mean(X, axis=0)
        sum_dist_mean = np.sum(np.linalg.norm(X - x_mean, axis=1) ** 2)

        # Initialize cluster weight eta using Eq. (3):
        eta = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            d2 = np.linalg.norm(X - centers[i], axis=1) ** 2
            numerator = np.sum((m[:, i] ** self.p) * d2)
            denominator = np.sum(m[:, i] ** self.p) + 1e-8
            eta[i] = numerator / denominator

        prev_obj = np.inf

        for it in range(self.max_iter):
            # Update sample weights gamma using Eq. (9)
            gamma = np.zeros((n, self.n_clusters))
            for i in range(self.n_clusters):
                for j in range(n):
                    d2_ji = np.linalg.norm(X[j] - centers[i]) ** 2
                    global_avg = sum_dist_mean / n
                    gamma[j, i] = np.exp(global_avg - d2_ji)

            # Update membership matrix m using Eq. (11)
            for j in range(n):
                for i in range(self.n_clusters):
                    d2_ji = np.linalg.norm(X[j] - centers[i]) ** 2 + 1e-8
                    denom_sum = 0.0
                    for k in range(self.n_clusters):
                        d2_jk = np.linalg.norm(X[j] - centers[k]) ** 2 + 1e-8
                        ratio = ((1 - gamma[j, i]) * d2_ji) / ((1 - gamma[j, k]) * d2_jk)
                        denom_sum += ratio ** (1.0 / (self.p - 1))
                    m[j, i] = 1.0 / (denom_sum + 1e-8)

            # Update typicality matrix t using Eq. (12)
            for j in range(n):
                for i in range(self.n_clusters):
                    d2_ji = np.linalg.norm(X[j] - centers[i]) ** 2 + 1e-8
                    num = self.lambda1 * eta[i] + self.lambda2 * b[j] * d2_ji * f[j, i]
                    den = self.lambda1 * gamma[j, i] * d2_ji + self.lambda1 * eta[i] + self.lambda2 * b[j] * d2_ji + 1e-8
                    t[j, i] = num / den

            # Update cluster centers using Eq. (13)
            z = np.zeros((n, self.n_clusters))
            for i in range(self.n_clusters):
                for j in range(n):
                    term1 = self.lambda1 * ((1 - gamma[j, i]) * (m[j, i] ** self.p) + gamma[j, i] * (t[j, i] ** self.q))
                    term2 = self.lambda2 * b[j] * ((t[j, i] - f[j, i]) ** self.q)
                    z[j, i] = term1 + term2
                numerator = np.sum(z[:, i].reshape(-1, 1) * X, axis=0)
                denominator = np.sum(z[:, i]) + 1e-8
                centers[i] = numerator / denominator

            # Update eta for each cluster using Eq. (3) again.
            for i in range(self.n_clusters):
                d2 = np.linalg.norm(X - centers[i], axis=1) ** 2
                numerator = np.sum((m[:, i] ** self.p) * d2)
                denominator = np.sum(m[:, i] ** self.p) + 1e-8
                eta[i] = numerator / denominator

            # Compute the objective function value J using Eq. (7)
            obj = 0.0
            for i in range(self.n_clusters):
                for j in range(n):
                    d2_ji = np.linalg.norm(X[j] - centers[i]) ** 2
                    term1 = (1 - gamma[j, i]) * (m[j, i] ** self.p) + gamma[j, i] * (t[j, i] ** self.q)
                    term2 = (1 - t[j, i]) ** self.q
                    term3 = b[j] * ((t[j, i] - f[j, i]) ** self.q) * d2_ji
                    obj += self.lambda1 * (term1 * d2_ji) + self.lambda1 * eta[i] * term2 + self.lambda2 * term3

            print(f"Iteration {it}, objective: {obj:.6f}")

            if abs(prev_obj - obj) < self.epsilon:
                print("Convergence reached at iteration", it)
                break
            prev_obj = obj

        # Save the resulting parameters
        self.centers_ = centers
        self.membership_ = m
        self.typicality_ = t
        self.eta_ = eta
        self.gamma_ = gamma
        return self

if __name__ == "__main__":
    # Generate synthetic data for demonstration
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=10)
    
    # Assume only a small fraction of points are labeled (e.g., 10%).
    labels = [int(label) if np.random.rand() < 0.1 else None for label in y_true]
    
    # Fit the model
    model = WSPFCM(n_clusters=3, lambda1=1.0, lambda2=1.0, p=2.0, q=2.0,
                   max_iter=50, epsilon=1e-4, random_state=42)
    model.fit(X, labels)
    
    # Assign each data point to the cluster with the highest membership value.
    cluster_assignment = np.argmax(model.membership_, axis=1)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_assignment, cmap='viridis', s=30, alpha=0.7)
    plt.scatter(model.centers_[:, 0], model.centers_[:, 1], color='red', marker='x', s=150, linewidths=3)
    plt.title("WSPFCM Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
