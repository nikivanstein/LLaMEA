import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitnesses = []
        self.selection_prob = 0.45
        self.crossover_prob = 0.45
        self.mutation_prob = 0.01

    def __call__(self, func):
        # Evaluate the function using the given budget
        func_evaluations = self.budget(func)
        
        # Generate a random initial population
        self.population = [func_evaluations] * self.dim
        np.random.shuffle(self.population)

        # Initialize the current best solution
        self.current_best = self.population[0]

        # Iterate until the budget is exhausted
        while len(self.population) < self.budget:
            # Select the next individual using the selection probability
            next_individual = self.select_next_individual()

            # Perform crossover (recombination) on the selected individual
            next_individual = self.crossover(next_individual)

            # Perform mutation on the selected individual
            next_individual = self.mutate(next_individual)

            # Evaluate the new individual using the function
            func_evaluations = self.budget(func)

            # Update the current best solution if necessary
            if func_evaluations < self.current_best.eval():
                self.current_best = next_individual

            # Store the fitness of the new individual
            self.fitnesses.append(func_evaluations)

            # Update the population
            self.population.append(next_individual)

        # Return the current best solution
        return self.current_best

    def select_next_individual(self):
        # Select the next individual using the selection probability
        return random.choices(self.population, weights=self.fitnesses, k=1)[0]

    def crossover(self, individual):
        # Perform crossover (recombination) on the selected individual
        crossover_point = random.randint(1, self.dim - 1)
        child = individual[:crossover_point] + individual[crossover_point + 1:]
        return child

    def mutate(self, individual):
        # Perform mutation on the selected individual
        mutation_point = random.randint(1, self.dim - 1)
        if random.random() < self.mutation_prob:
            individual[mutation_point] += random.uniform(-1, 1)
        return individual