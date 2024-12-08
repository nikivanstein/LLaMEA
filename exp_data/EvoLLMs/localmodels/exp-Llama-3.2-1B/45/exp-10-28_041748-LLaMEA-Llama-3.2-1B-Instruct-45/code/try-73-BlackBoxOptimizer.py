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

def l2(f, budget):
    """L2-norm fitness function"""
    return np.sum(np.abs(f)) / budget

def mutation(individual, mutation_rate):
    """Random mutation function"""
    if random.random() < mutation_rate:
        return random.uniform(self.search_space[0], self.search_space[1])
    return individual

def crossover(parent1, parent2, crossover_rate):
    """Crossover function"""
    child = (parent1 + parent2) / 2
    if random.random() < crossover_rate:
        return mutation(child, mutation_rate)
    return child

def bbob_optimization(budget, dim):
    """Black Box Optimization using Evolutionary Algorithm"""
    optimizer = BlackBoxOptimizer(budget, dim)
    best_individual = optimizer(func)
    best_fitness = l2(best_individual, budget)
    print(f"Best Individual: {best_individual}, Best Fitness: {best_fitness}")

# Run the optimization algorithm
bbob_optimization(1000, 10)