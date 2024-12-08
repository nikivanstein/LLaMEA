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
        self.logger = None

    def __call__(self, func):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        if self.logger is not None:
            self.logger.update('New population size: ', len(top_individuals))
        
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            
            if random.random() < self.mutation_rate:
                if self.logger is not None:
                    self.logger.update('Mutation rate: ', self.mutation_rate)
                child = random.uniform(self.search_space[0], self.search_space[1])
            
            new_population.append(child)
        
        # Replace the old population with the new one
        if self.logger is not None:
            self.logger.update('Old population size: ', len(self.population))
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Return the best individual
        if self.logger is not None:
            self.logger.update('Best individual: ', np.argmax(new_func_evaluations))
        best_individual = np.argmax(new_func_evaluations)
        
        return new_population[best_individual]

class MutationExp:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1
        self.logger = None

    def __call__(self, func):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        if self.logger is not None:
            self.logger.update('New population size: ', len(top_individuals))
        
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            
            if random.random() < self.mutation_rate:
                if self.logger is not None:
                    self.logger.update('Mutation rate: ', self.mutation_rate)
                child = random.uniform(self.search_space[0], self.search_space[1])
            
            new_population.append(child)
        
        # Replace the old population with the new one
        if self.logger is not None:
            self.logger.update('Old population size: ', len(self.population))
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Return the best individual
        if self.logger is not None:
            self.logger.update('Best individual: ', np.argmax(new_func_evaluations))
        best_individual = np.argmax(new_func_evaluations)
        
        return new_population[best_individual]

# Initialize the optimizer
optimizer = BlackBoxOptimizer(1000, 10)
# Initialize the mutation exp
mutation_exp = MutationExp(1000, 10)

# Run the optimizer for 1000 iterations
for _ in range(1000):
    func = lambda x: np.sin(x)
    best_individual = optimizer.__call__(func)
    print(f'Best individual: {best_individual}')
    print(f'Best fitness: {np.max(np.array([np.sin(x) for x in np.random.uniform(-10, 10, 1000)]))}')
    print()