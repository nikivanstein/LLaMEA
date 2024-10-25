import numpy as np
import matplotlib.pyplot as plt

class NoisyBlackBoxOptimizer:
    def __init__(self, budget, dim, max_iter=1000):
        self.budget = budget
        self.dim = dim
        self.max_iter = max_iter
        self.explore_eviction = False
        self.current_dim = 0
        self.func = None

    def __call__(self, func):
        if self.explore_eviction:
            # Hierarchical clustering to select the best function to optimize
            cluster_labels = np.argpartition(func, self.current_dim)[-1]
            self.explore_eviction = False
            return func
        else:
            # Hierarchical clustering to select the best function to optimize
            cluster_labels = np.argpartition(func, self.current_dim)[-1]
            self.explore_eviction = False
            self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
            self.current_dim += 1
            if self.budget == 0:
                break
            self.budget -= 1
            return self.func

    def __str__(self):
        return f"NoisyBlackBoxOptimizer: Hierarchical clustering to select the best function to optimize"

    def __str__(self):
        return f"Dim: {self.dim}, Budget: {self.budget}, Explored Eviction: {self.explore_eviction}, Current Dim: {self.current_dim}"

# Example usage
if __name__ == "__main__":
    optimizer = NoisyBlackBoxOptimizer(1000, 10)
    func = lambda x: np.sin(x)
    best_func = None
    best_fitness = float('inf')
    for _ in range(100):
        new_individual = optimizer.func(np.random.uniform(-5.0, 5.0, 10))
        fitness = np.mean(np.abs(new_individual - func(new_individual)))
        print(f"Individual: {new_individual}, Fitness: {fitness}")
        if fitness < best_fitness:
            best_func = new_individual
            best_fitness = fitness
    print(f"Best Function: {best_func}, Best Fitness: {best_fitness}")