import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population = []

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
        # Refine the strategy by changing the lines of the individual
        lines = individual.split('\n')
        new_lines = []
        for line in lines:
            if random.random() < 0.15:  # 15% chance of changing a line
                new_lines.append(line +'| |')
            else:
                new_lines.append(line)
        individual = '\n'.join(new_lines)
        return individual

    def crossover(self, parent1, parent2):
        # Perform crossover to combine the best of both parents
        child = ''
        for line in parent1.split('\n'):
            if random.random() < 0.5:  # 50% chance of taking a line from parent1
                child += line + '\n'
            else:
                child += parent2.split('\n')[random.randint(0, len(parent2.split('\n'))-1)] + '\n'
        child = child.rstrip('\n')  # Remove trailing newline
        return child

    def evolve(self, population):
        # Evolve the population by applying mutation and crossover
        new_population = []
        for _ in range(10):  # Evolve for 10 generations
            for individual in population:
                individual = self.mutate(individual)
                individual = self.crossover(individual, self.population[0])
                individual = self.crossover(individual, self.population[1])
                new_population.append(individual)
        population = new_population
        return population

    def get_best(self, population):
        # Return the best individual in the population
        best_individual = max(population, key=self.evaluate_fitness)
        return best_individual

# Example usage:
budget = 100
dim = 10
optimizer = BlackBoxOptimizer(budget, dim)
func = lambda x: x**2  # Simple black box function
best_individual = optimizer.get_best(optimizer.evolve([optimizer.get_best([func]) for _ in range(10)]))
print("Best individual:", best_individual)
print("Best function value:", func(best_individual))