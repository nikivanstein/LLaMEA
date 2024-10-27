import random
import numpy as np

class DynamicEvolutionaryAlgorithm:
    def __init__(self, budget, dim, mutation_rate):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutation_rate = mutation_rate

    def __call__(self, func, problem, population_size, generations):
        # Initialize the population with random individuals
        population = self.generate_population(population_size)

        # Evaluate the fitness of each individual
        fitnesses = [self.evaluate_fitness(individual, func, problem) for individual in population]

        # Select the fittest individuals
        fittest_individuals = self.select_fittest(population, fitnesses)

        # Create a new generation of individuals
        new_population = self.generate_new_generation(population_size, fittest_individuals, fitnesses)

        # Evaluate the fitness of each individual in the new population
        fitnesses = [self.evaluate_fitness(individual, func, problem) for individual in new_population]

        # Update the population based on the performance of the new generation
        self.update_population(new_population, fitnesses, population_size, generations)

        # Return the fittest individual in the new population
        return self.select_fittest(new_population, fitnesses)

    def generate_population(self, population_size):
        # Generate a population of random individuals
        return [np.random.choice(self.search_space, size=self.dim) for _ in range(population_size)]

    def evaluate_fitness(self, individual, func, problem):
        # Evaluate the fitness of the individual
        value = func(individual)
        return value

    def select_fittest(self, population, fitnesses):
        # Select the fittest individuals
        fittest_individuals = [individual for _, individual in sorted(zip(fitnesses, population), reverse=True)]
        return fittest_individuals

    def generate_new_generation(self, population_size, fittest_individuals, fitnesses):
        # Create a new generation of individuals
        new_population = [individual for individual in population_size * fittest_individuals[:len(fittest_individuals) // 2] + fittest_individuals[len(fittest_individuals) // 2:]
                        for _ in range(population_size - len(fittest_individuals) // 2)]
        return new_population

    def update_population(self, new_population, fitnesses, population_size, generations):
        # Update the population based on the performance of the new generation
        for individual, fitness in zip(new_population, fitnesses):
            if fitness > 0.5:
                individual = individual * self.mutation_rate
                individual = np.clip(individual, -5.0, 5.0)
            else:
                individual = individual * (1 - self.mutation_rate)
                individual = np.clip(individual, -5.0, 5.0)
        return new_population

# One-line description: "Dynamic Evolutionary Algorithm: A novel metaheuristic algorithm that efficiently solves black box optimization problems by dynamically adjusting the strategy of the evolutionary algorithm based on the performance of the current solution"

# Example usage:
budget = 1000
dim = 5
mutation_rate = 0.01
problem = RealSingleObjectiveProblem(1, "Sphere", iid=1, dim=dim)
optimizer = DynamicEvolutionaryAlgorithm(budget, dim, mutation_rate)
best_solution = optimizer(__call__, problem, 100, 100)
print(best_solution)