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
        self.population = None
        self.population_history = None
        self.population_best = None

    def __call__(self, func):
        if self.explore_eviction:
            # Hierarchical clustering to select the best function to optimize
            cluster_labels = np.argpartition(func, self.current_dim)[-1]
            self.explore_eviction = False
            return func
        else:
            # Gradient descent with hierarchical clustering for efficient exploration-ejection
            while self.budget > 0 and self.current_dim < self.dim:
                if self.explore_eviction:
                    # Select the best function to optimize using hierarchical clustering
                    cluster_labels = np.argpartition(func, self.current_dim)[-1]
                    self.explore_eviction = False
                else:
                    # Perform gradient descent with hierarchical clustering for efficient exploration-ejection
                    if self.current_dim == 0:
                        # Gradient descent without hierarchical clustering
                        self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
                    else:
                        # Hierarchical clustering to select the best function to optimize
                        cluster_labels = np.argpartition(func, self.current_dim)[-1]
                        self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels == cluster_labels[self.current_dim]])
                        self.current_dim += 1
                        if self.budget == 0:
                            break
                self.budget -= 1
            return self.func

    def evolve_population(self, population):
        # Select the fittest individuals for the next generation
        self.population = population
        self.population_history = [self.func(x) for x in population]
        self.population_best = self.func(np.array([x for x in self.population if x not in self.population_history]))

    def __repr__(self):
        return f"NoisyBlackBoxOptimizer(budget={self.budget}, dim={self.dim}, max_iter={self.max_iter}, explore_eviction={self.explore_eviction}, current_dim={self.current_dim}, func={self.func}, population={self.population}, population_history={self.population_history}, population_best={self.population_best})"

# One-line description: Hierarchical black box optimization using evolutionary strategies to find the optimal function in a noisy black box problem.