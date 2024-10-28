import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        # Generate a population of random solutions
        return [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

    def fitness(self, func, solution):
        # Evaluate the function at the given solution
        return func(solution)

    def __call__(self, func, population_size, budget):
        # Evaluate the function at each solution in the population
        for _ in range(budget):
            # Select the fittest solutions
            fittest_solutions = sorted(self.population, key=self.fitness, reverse=True)[:self.population_size // 2]

            # Select a random subset of solutions to mutate
            mutation_solutions = random.sample(fittest_solutions, self.population_size - self.population_size // 2)

            # Mutate the selected solutions
            mutated_solutions = []
            for solution in mutation_solutions:
                if random.random() < self.mutation_rate:
                    # Randomly change a single element in the solution
                    mutated_solution = solution.copy()
                    mutated_solution[random.randint(0, self.dim - 1)] = random.uniform(-5.0, 5.0)
                    mutated_solutions.append(mutated_solution)

            # Evaluate the function at the mutated solutions
            new_fittest_solutions = sorted(mutated_solutions, key=self.fitness, reverse=True)[:self.population_size // 2]

            # Replace the least fit solutions with the new fittest solutions
            self.population = new_fittest_solutions

        # Return the fittest solution
        return self.population[0]

    def run(self, func, population_size, budget):
        # Run the algorithm until it converges
        while self.fitness(func, self.population[0]) > 1e-6:
            self.population = self.__call__(func, population_size, budget)

# Description: Black Box Optimization using Genetic Algorithm
# Code: 