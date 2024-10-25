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
            # Genetic algorithm with evolutionary strategy
            self.func = self.ga_with_evolutionary_strategy(func)
            return self.func

    def ga_with_evolutionary_strategy(self, func):
        population_size = 100
        population = self.initialize_population(func, population_size)
        fitnesses = np.zeros(len(population))
        for _ in range(self.max_iter):
            # Selection
            fitnesses = self.selection(population, fitnesses)
            # Crossover
            population = self.crossover(population, fitnesses)
            # Mutation
            population = self.mutation(population, fitnesses)
        return population

    def initialize_population(self, func, population_size):
        return np.random.uniform(-5.0, 5.0, (population_size, self.dim))

    def selection(self, population, fitnesses):
        # Simple tournament selection
        selection = np.zeros(len(population))
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                if fitnesses[i] > fitnesses[j]:
                    selection[i] = 1
                    selection[j] = 0
        return selection

    def crossover(self, population, fitnesses):
        # Hierarchical crossover
        crossover_points = np.arange(len(population))
        offspring = np.zeros((len(population) * 2, self.dim))
        for i in range(len(population)):
            parent1 = population[crossover_points[i]]
            parent2 = population[crossover_points[i + 1]]
            if self.explore_eviction:
                # Select the best function to optimize using hierarchical clustering
                cluster_labels = np.argpartition(func, i)[-1]
                parent1 = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels == cluster_labels[i]])
                parent2 = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels == cluster_labels[i + 1]])
            else:
                # Perform hierarchical crossover without hierarchical clustering
                if self.current_dim == 0:
                    # Simple crossover
                    offspring = np.concatenate((parent1[:self.dim], parent2[:self.dim]))
                else:
                    # Hierarchical crossover
                    cluster_labels = np.argpartition(func, i)[-1]
                    offspring = np.concatenate((parent1[:self.dim], parent2[:self.dim] if cluster_labels == cluster_labels[i] else parent2[:self.dim]))
                    offspring = np.concatenate((offspring[:self.dim], parent1[self.dim:] if cluster_labels == cluster_labels[i + 1] else parent1[self.dim:]))
            offspring = self.func(offspring)
            fitnesses[i] = np.array([func(x) for x in offspring])
        return offspring

    def mutation(self, population, fitnesses):
        # Simple mutation
        mutation_rate = 0.1
        for i in range(len(population)):
            if np.random.rand() < mutation_rate:
                fitnesses[i] = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
        return population, fitnesses

# Example usage
noisy_black_box_optimizer = NoisyBlackBoxOptimizer(budget=100, dim=10)
func = lambda x: np.sin(x)
optimized_func = noisy_black_box_optimizer(func)
print("Optimized function:", optimized_func)