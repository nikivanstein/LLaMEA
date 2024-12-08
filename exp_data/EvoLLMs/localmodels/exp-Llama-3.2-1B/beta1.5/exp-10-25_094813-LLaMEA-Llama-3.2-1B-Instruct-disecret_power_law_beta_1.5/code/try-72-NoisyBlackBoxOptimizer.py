import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

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

    def func(self, x):
        return np.array([func(x) for func in self.func])

def evaluate_fitness(individual, func, budget):
    new_individual = individual
    for _ in range(budget):
        new_individual = func(new_individual)
    return new_individual

def mutation_exp(individual, func, mutation_rate):
    new_individual = individual
    for _ in range(int(np.random.rand() * mutation_rate)):
        new_individual = func(new_individual)
    return new_individual

# Test the optimizer
budget = 1000
dim = 10
max_iter = 1000
optimizer = NoisyBlackBoxOptimizer(budget, dim, max_iter)

# Initial population
initial_population = [np.random.uniform(-5.0, 5.0, dim) for _ in range(100)]

# Evaluate the initial population
fitnesses = [evaluate_fitness(individual, func, budget) for individual, func in zip(initial_population, [func for func in optimizer.func])]

# Print the initial population and fitnesses
print("Initial Population:")
for i, individual in enumerate(initial_population):
    print(f"Individual {i+1}: {individual}, Fitness: {fitnesses[i]}")

# Select the best individual
best_individual = initial_population[np.argmax(fitnesses)]

# Print the best individual
print(f"\nBest Individual: {best_individual}")

# Update the population using the optimizer
new_population = [mutation_exp(individual, func, mutation_rate) for individual, func in zip(initial_population, [func for func in optimizer.func])]

# Evaluate the new population
fitnesses = [evaluate_fitness(individual, func, budget) for individual, func in zip(new_population, [func for func in optimizer.func])]

# Print the new population and fitnesses
print("\nNew Population:")
for i, individual in enumerate(new_population):
    print(f"Individual {i+1}: {individual}, Fitness: {fitnesses[i]}")

# Print the updated population and fitnesses
print("\nUpdated Population:")
for i, individual in enumerate(new_population):
    print(f"Individual {i+1}: {individual}, Fitness: {fitnesses[i]}")