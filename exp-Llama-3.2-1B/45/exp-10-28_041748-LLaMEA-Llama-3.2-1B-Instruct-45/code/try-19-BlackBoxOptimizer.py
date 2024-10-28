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
        return new_population[best_individual]

    def mutate(self, individual):
        # Randomly select a mutation point
        mutation_point = random.randint(0, self.dim)
        
        # Swap the element at the mutation point with a random element from the search space
        if random.random() < self.mutation_rate:
            individual[mutation_point], random.choice(self.search_space) = random.choice(self.search_space), individual[mutation_point]
        
        # Return the mutated individual
        return individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(0, self.dim)
        
        # Create a child by combining the two parents
        child = (parent1 + parent2) / 2
        
        # Return the child
        return child

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 