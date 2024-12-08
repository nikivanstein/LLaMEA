import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        # Define the mutation rate
        mutation_rate = 0.01

        # Initialize the population with random individuals
        population = [self.generate_random_individual() for _ in range(100)]

        for _ in range(1000):  # Run for 1000 generations
            # Evaluate the fitness of each individual
            fitness = [func(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = [individual for individual, fitness in zip(population, fitness) if fitness == max(fitness)]

            # Select a random subset of individuals to mutate
            mutation_indices = random.sample(range(len(fittest_individuals)), int(self.budget * mutation_rate))

            # Mutate the selected individuals
            mutated_individuals = [fittest_individuals[i] for i in mutation_indices]

            # Evaluate the fitness of the mutated individuals
            mutated_fitness = [func(individual) for individual in mutated_individuals]

            # Replace the fittest individuals with the mutated ones
            fittest_individuals = [individual for individual, fitness in zip(population, mutated_fitness) if fitness == max(mutated_fitness)]

            # Update the population
            population = fittest_individuals

            # Check if the budget is reached
            if len(population) == self.budget:
                # If the budget is reached, return the best individual found so far
                return max(population)
        # If the budget is not reached, return the best individual found so far
        return max(population)

    def generate_random_individual(self):
        return (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))

# Define the function to save the population
def save_population(population, algorithm_name):
    np.save(f"currentexp/{algorithm_name}-{len(population)}.npy", population)

# Define the function to evaluate the fitness of an individual
def evaluate_fitness(individual, func):
    return func(individual)

# Define the function to mutate an individual
def mutate(individual):
    return (individual[0] + random.uniform(-1.0, 1.0), individual[1] + random.uniform(-1.0, 1.0))

# Define the function to run the optimization algorithm
def run_optimization_algorithm(func, algorithm_name, budget, dim):
    optimizer = BlackBoxOptimizer(budget, dim)
    return optimizer()

# Example usage:
func = lambda x: x**2
algorithm_name = "Novel Metaheuristic Algorithm for Black Box Optimization"
budget = 100
dim = 2

best_individual = run_optimization_algorithm(func, algorithm_name, budget, dim)
print(f"The best individual found is: {best_individual}")