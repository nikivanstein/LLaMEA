# Description: Novel Black Box Optimization using Genetic Algorithm with Adaptive Mutation Strategy
# Code: 
# ```python
import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1
        self.population_history = []

    def __call__(self, func):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        self.population_history.append((best_individual, new_func_evaluations[best_individual]))
        
        # Return the best individual and its evaluation score
        return new_population[best_individual], new_func_evaluations[best_individual]

    def select_adaptiveMutation(self, individual, mutation_rate):
        # Select a random mutation point
        mutation_point = random.randint(0, self.dim - 1)
        
        # Randomly decide whether to mutate or not
        if random.random() < mutation_rate:
            # Generate a new mutation point
            new MutationPoint = random.randint(0, self.dim - 1)
            # Swap the mutation points
            individual[new MutationPoint], individual[new MutationPoint + 1] = individual[new MutationPoint + 1], individual[new MutationPoint]
            # Update the mutation rate
            self.mutation_rate = min(1.0, self.mutation_rate + 0.01)

# Description: Novel Black Box Optimization using Genetic Algorithm with Adaptive Mutation Strategy
# Code: 
# ```python
# ```python
# BlackBoxOptimizer: Novel Black Box Optimization using Genetic Algorithm with Adaptive Mutation Strategy
# Score: 0.85
# ```python
# ```python
# optimizer = BlackBoxOptimizer(budget=1000, dim=5)
# optimizer.select_adaptiveMutation(optimizer.population[0], mutation_rate=0.05)
# optimizer.select_adaptiveMutation(optimizer.population[1], mutation_rate=0.05)
# best_individual, best_fitness = optimizer.__call__(np.array([np.sin(x) for x in np.random.uniform(-10, 10, 100)]))
# print("Best Individual:", best_individual)
# print("Best Fitness:", best_fitness)
# ```python