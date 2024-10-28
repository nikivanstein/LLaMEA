# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random

class HyperGraphOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._build_graph()
        self.population = self._init_population()

    def _build_graph(self):
        # Create a graph where each node represents a dimension and each edge represents a possible hyper-parameter combination
        graph = {}
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                graph[(i, j)] = 1
        return graph

    def _init_population(self):
        # Initialize the population with random hyper-parameter combinations
        population = []
        for _ in range(1000):
            combination = tuple(random.choices(range(self.dim), k=self.dim))
            population.append(combination)
        return population

    def _select_parent(self, population):
        # Select the parent using the tournament selection algorithm
        parent = random.choice(population)
        for _ in range(10):
            child = tuple(random.choices(population, k=self.dim))
            if child!= parent:
                parent = child
                break
        return parent

    def _crossover(self, parent1, parent2):
        # Perform crossover to create a new child
        child = tuple()
        for i in range(self.dim):
            if random.random() < 0.5:
                child += parent1[i]
            else:
                child += parent2[i]
        return child

    def _mutate(self, parent):
        # Perform mutation on the parent
        mutated = parent.copy()
        if random.random() < 0.1:
            index = random.randint(0, self.dim - 1)
            mutated[index] += random.uniform(-1, 1)
        return mutated

    def __call__(self, func):
        # Optimize the black box function using the Hyper-Graph Optimizer
        for _ in range(self.budget):
            # Select the parents using the tournament selection algorithm
            parent1 = self._select_parent(self.population)
            parent2 = self._select_parent(self.population)

            # Perform crossover and mutation to create new children
            child1 = self._crossover(parent1, parent2)
            child2 = self._mutate(parent1)

            # Evaluate the function for each child
            results1 = [func(child1) for child1 in [child1, child2]]
            results2 = [func(child2) for child2 in [child1, child2]]
            # Find the combination that maximizes the function value
            max_index1 = np.argmax(results1)
            max_index2 = np.argmax(results2)
            # Update the graph and the optimized function
            self.graph[max_index1] = max_index1
            self.graph[max_index2] = max_index2
            self.budget -= 1
            # If the budget is exhausted, break the loop
            if self.budget == 0:
                break
        # Return the optimized function
        return self.budget, self.graph

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# Using a graph-based representation and a tournament selection algorithm with crossover and mutation to optimize black box functions.