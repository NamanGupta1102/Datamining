import numpy as np
import matplotlib.pyplot as plt

def auto_elbow(x, y, plot=True):
    """
    Estimate the optimal number of clusters (k) from an elbow plot.
    
    This function normalizes the input x (number of clusters) and y (TWSD or inertia)
    values to [0,1] and computes the perpendicular distance from each point to the
    line joining the first and last points. The optimal k is taken as the x-value
    corresponding to the maximum distance.
    
    Parameters:
    -----------
    x : array-like, shape (n_points,)
        The k-values (e.g., number of clusters tried).
    y : array-like, shape (n_points,)
        The corresponding evaluation metric (e.g., TWSD/inertia).
    plot : bool, optional (default=True)
        If True, plots the elbow curve with the detected elbow point highlighted.
    
    Returns:
    --------
    k_optimal : numeric
        The estimated optimal number of clusters.
    distances : ndarray
        Array of perpendicular distances for each point.
    """
    x = np.array(x)
    y = np.array(y)
    
    # Normalize x and y to the interval [0, 1]
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    
    # Define the first (O) and last (Q) points of the normalized curve
    O = np.array([x_norm[0], y_norm[0]])
    Q = np.array([x_norm[-1], y_norm[-1]])
    
    # Compute perpendicular distances from each point to the line OQ
    distances = []
    for i in range(len(x_norm)):
        P = np.array([x_norm[i], y_norm[i]])
        # Perpendicular distance = area of parallelogram / base length
        distance = np.abs(np.cross(Q - O, P - O)) / np.linalg.norm(Q - O)
        distances.append(distance)
    distances = np.array(distances)
    
    # The elbow is at the point with the maximum distance
    elbow_idx = np.argmax(distances)
    k_optimal = x[elbow_idx]
    
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'bo-', label='Elbow Curve')
        plt.plot([x[0], x[-1]], [y[0], y[-1]], 'r--', label='Line connecting endpoints')
        plt.plot(x[elbow_idx], y[elbow_idx], 'ro', markersize=12, label=f'Elbow at k={k_optimal}')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('TWSD / Inertia')
        plt.title('Automatic Elbow Detection')
        plt.legend()
        plt.show()
    
    return k_optimal, distances

# Example usage:
if __name__ == "__main__":
    # Suppose you run k-means for k=1..10 and record the TWSD (inertia) values.
    k_values = np.arange(1, 11)
    # Example TWSD values (a typical decreasing curve with a bend)
    inertia = np.array([1000, 800, 650, 540, 500, 480, 470, 465, 460, 455])
    
    optimal_k, dists = auto_elbow(k_values, inertia)
    print("Estimated optimal number of clusters:", optimal_k)
