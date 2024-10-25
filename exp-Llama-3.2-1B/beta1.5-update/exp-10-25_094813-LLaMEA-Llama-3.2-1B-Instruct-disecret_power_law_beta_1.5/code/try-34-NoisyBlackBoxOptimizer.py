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
        self.population_history = []
        self.fitness_history = []
        self.best_solution = None
        self.best_fitness = np.inf
        self.population_size = 100

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

    def select_solution(self, individual):
        # Hierarchical clustering-based selection
        cluster_labels = np.argpartition(individual, self.current_dim)[-1]
        self.current_dim += 1
        if self.budget == 0:
            break
        if self.explore_eviction:
            # Select the best individual to optimize using hierarchical clustering
            best_individual = np.array([func(x) for func in self.func if cluster_labels == cluster_labels[self.current_dim]])
            self.explore_eviction = False
        else:
            # Select the best individual to optimize using gradient descent
            best_individual = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels == cluster_labels[self.current_dim]])
            self.explore_eviction = False
        return best_individual

    def evaluate_fitness(self, individual):
        return np.array([func(individual) for func in self.func])

    def fitness(self, individual):
        return np.array([func(individual) for func in self.func])

    def mutate(self, individual):
        # Hierarchical clustering-based mutation
        if self.current_dim == 0:
            # Gradient descent without hierarchical clustering
            mutated_individual = np.random.uniform(-5.0, 5.0, self.dim)
        else:
            # Hierarchical clustering-based mutation
            mutated_individual = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if np.argpartition(func(x), self.current_dim - 1)[-1] == np.argpartition(func(x), self.current_dim)[-1]])
        return mutated_individual

    def train(self, num_generations):
        # Hierarchical clustering-based training
        for generation in range(num_generations):
            # Select the best individual to optimize
            individual = self.select_solution(self.population)
            # Evaluate the fitness of the individual
            fitness = self.evaluate_fitness(individual)
            # Update the population
            self.population_history.append(individual)
            self.fitness_history.append(fitness)
            # Mutate the individual
            mutated_individual = self.mutate(individual)
            # Evaluate the fitness of the mutated individual
            fitness = self.evaluate_fitness(mutated_individual)
            # Update the population
            self.population_history.append(mutated_individual)
            self.fitness_history.append(fitness)
            # Select the best individual to optimize
            individual = self.select_solution(self.population)
            # Evaluate the fitness of the individual
            fitness = self.evaluate_fitness(individual)
            # Update the best solution
            self.best_solution = individual
            self.best_fitness = np.max(self.fitness_history)

    def plot_history(self):
        # Hierarchical clustering-based plotting
        plt.plot(self.fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.show()

# Description: Hierarchical Clustering-based Noisy Black Box Optimization
# Code: 
# ```python
noisy_black_box_optimizer = NoisyBlackBoxOptimizer(budget=1000, dim=10, max_iter=100)
noisy_black_box_optimizer.train(num_generations=100)
noisy_black_box_optimizer.plot_history()