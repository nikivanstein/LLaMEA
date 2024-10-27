import numpy as np
from scipy.optimize import minimize

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.01

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def select_solution(self):
        # Select the fittest individual based on the fitness
        fitnesses = [self.func(x) for x in self.search_space]
        self.fittest_individual = np.argmax(fitnesses)
        return self.fittest_individual

    def mutate(self, individual):
        # Randomly mutate the individual
        if np.random.rand() < self.mutation_rate:
            index = np.random.randint(0, len(individual))
            individual[index] = np.random.uniform(-5.0, 5.0)
        return individual

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        child = parent1[:len(parent2)//2] + parent2[len(parent2)//2:]
        return child

    def evolve(self, population_size):
        # Evolve the population using genetic algorithm
        while True:
            # Select the fittest individual
            self.fittest_individual = self.select_solution()
            # Mutate the fittest individual
            mutated_individual = self.mutate(self.fittest_individual)
            # Create two new parents by crossover
            parent1 = self.crossover(mutated_individual, mutated_individual)
            parent2 = self.crossover(mutated_individual, mutated_individual)
            # Add the new parents to the population
            population = [parent1, parent2, mutated_individual]
            # Evaluate the new population
            new_population = [self.func(x) for x in population]
            # Replace the old population with the new one
            self.population = population
            # Check if the population has reached the budget
            if len(new_population) == self.population_size:
                break
        return population

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Print the selected solution
print("Selected solution:", optimizer.select_solution())

# Evolve the population
population = optimizer.evolve(100)
print("Evolved population:", population)