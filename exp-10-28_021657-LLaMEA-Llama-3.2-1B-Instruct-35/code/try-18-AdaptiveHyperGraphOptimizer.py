import numpy as np
import random

class AdaptiveHyperGraphOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._build_graph()
        self.population = self._initialize_population()
        self.best_solution = None
        self.best_score = float('inf')
        self.best_individual = None

    def _build_graph(self):
        # Create a graph where each node represents a dimension and each edge represents a possible hyper-parameter combination
        graph = {}
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                graph[(i, j)] = 1
        return graph

    def _initialize_population(self):
        # Initialize the population with random hyper-parameter combinations
        population = []
        for _ in range(self.budget):
            individual = tuple(random.randint(-5.0, 5.0) for _ in range(self.dim))
            population.append(individual)
        return population

    def __call__(self, func):
        # Optimize the black box function using the Adaptive HyperGraph Optimizer
        while True:
            # Generate all possible hyper-parameter combinations
            combinations = self._generate_combinations()
            # Evaluate the function for each combination
            results = [func(combination) for combination in combinations]
            # Find the combination that maximizes the function value
            max_index = np.argmax(results)
            # Update the graph and the optimized function
            self.graph[max_index] = max_index
            self.budget -= 1
            # If the budget is exhausted, break the loop
            if self.budget == 0:
                break
            # Select the best individual based on the probability of convergence
            probability = 0.35
            if self.budget > 0:
                best_individual = random.choices(self.population, weights=[self.graph[i] for i in self.population], k=1)[0]
            else:
                best_individual = self.population[0]
            # Update the best solution and score
            if self.budget > 0:
                self.best_individual = best_individual
                self.best_score = max(self.best_score, func(best_individual))
                if func(best_individual) > self.best_score:
                    self.best_individual = best_individual
                    self.best_score = func(best_individual)
            # Update the population
            self.population = [individual for individual in self.population if self.graph[i] == i for i in range(self.dim)]
        # Return the optimized function
        return self.best_individual, self.best_score

    def _generate_combinations(self):
        # Generate all possible hyper-parameter combinations
        combinations = []
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                combinations.append((i, j))
        return combinations

# One-line description with the main idea
# Adaptive HyperGraph Optimizer: A novel metaheuristic algorithm that adapts its search strategy based on the probability of convergence to optimize black box optimization problems.
# Code: 