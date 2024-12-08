import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.init_population()

    def init_population(self):
        # Initialize the population with random solutions
        return [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(self.population_size)]

    def __call__(self, func):
        # Evaluate the black box function func with the current population
        evaluations = []
        for _ in range(self.budget):
            for solution in self.population:
                evaluation = func(solution)
                evaluations.append(evaluation)
        # Select the fittest solutions
        fitness = [evaluation for evaluation in evaluations]
        idx = np.argsort(fitness)[::-1]
        self.population = [self.population[i] for i in idx]
        return self.population

    def select_solution(self):
        # Refine the solution using the 0.45 rule
        if random.random() < 0.45:
            # If the solution is not good enough, try a better solution
            return self.population[0]
        else:
            # Otherwise, select the best solution from the current population
            return self.population[0]

    def mutate(self, solution):
        # Randomly swap two elements in the solution
        i, j = random.sample(range(self.dim), 2)
        solution[i], solution[j] = solution[j], solution[i]
        return solution

    def crossover(self, parent1, parent2):
        # Perform crossover between two parent solutions
        child = [solution for solution in parent1 if solution not in parent2]
        return child

    def __str__(self):
        return f"Population: {self.population}, Best Solution: {self.select_solution()}"