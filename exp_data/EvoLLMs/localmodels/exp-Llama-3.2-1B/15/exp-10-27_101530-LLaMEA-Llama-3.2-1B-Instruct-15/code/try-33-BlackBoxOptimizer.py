import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def mutate(self, individual):
        # Randomly perturb the individual
        mutated_individual = (individual[0] + random.uniform(-1.0, 1.0), individual[1] + random.uniform(-1.0, 1.0))
        # Ensure the mutated individual stays within the search space
        mutated_individual = (max(self.search_space[0], min(mutated_individual[0], self.search_space[1])), 
                             max(self.search_space[0], min(mutated_individual[1], self.search_space[1])))
        return mutated_individual

    def crossover(self, parent1, parent2):
        # Perform crossover between the two parents
        child = (parent1[0] + parent2[0] / 2, parent1[1] + parent2[1] / 2)
        return child

    def evolve(self, population_size, mutation_rate, crossover_rate):
        # Initialize the population
        population = [(self.evaluate_fitness(individual), individual) for individual in random.sample([self.evaluate_fitness(individual) for individual in population], population_size)]
        
        # Evolve the population
        for _ in range(100):
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=lambda x: x[0], reverse=True)[:int(population_size / 2)]
            
            # Create a new generation
            new_population = []
            for _ in range(population_size):
                # Select two parents
                parent1, parent2 = random.sample(fittest_individuals, 2)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutate
                if random.random() < mutation_rate:
                    child = self.mutate(child)
                
                # Add the child to the new population
                new_population.append(child)
            
            # Replace the old population with the new one
            population = new_population
        
        # Return the best individual in the new population
        return self.evaluate_fitness(max(population, key=lambda x: x[0])[1])

    def evaluate_fitness(self, func):
        # Evaluate the function at the individual
        return func(self.evaluate_fitness(func))

# Initialize the BlackBoxOptimizer
optimizer = BlackBoxOptimizer(100, 10)

# Initialize the best solution
best_solution = (optimizer.search_space[0], optimizer.search_space[1])

# Evaluate the best solution
best_func_value = optimizer(func, best_solution)
print(f"The best function value is {best_func_value}")

# Update the BlackBoxOptimizer
new_optimizer = BlackBoxOptimizer(100, 10)
best_new_solution = new_optimizer.evolve(100, 0.1, 0.1)
new_best_func_value = new_optimizer(func, best_new_solution)
print(f"The new best function value is {new_best_func_value}")