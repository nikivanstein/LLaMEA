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

    def __call__(self, func):
        if self.explore_eviction:
            # Hierarchical clustering to select the best function to optimize
            cluster_labels = np.argpartition(func, self.current_dim)[-1]
            self.explore_eviction = False
            return func
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
            return self.func

    def select_individual(self):
        if self.population is None:
            # Create a population of random individuals
            self.population = np.array([np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)])
        else:
            # Select the best individual from the population
            self.population = np.array([self.population[i] for i in range(len(self.population)) if i == np.argmax(self.population)])
        return self.population

    def mutate(self, individual):
        if self.explore_eviction:
            # Hierarchical clustering to select the best function to optimize
            cluster_labels = np.argpartition(individual, self.current_dim)[-1]
            self.explore_eviction = False
            # Perform mutation using hierarchical clustering
            self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels!= cluster_labels[self.current_dim]])
        else:
            # Perform mutation using gradient descent
            # Update individual with random mutation
            self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
        return self.func

    def __call__(self, func):
        # Call the optimization function
        return self.__call__(self.select_individual())

# Example usage:
optimizer = NoisyBlackBoxOptimizer(budget=100, dim=10, max_iter=1000)
func = lambda x: np.sin(x)
individual = optimizer.func(func)
print(individual)