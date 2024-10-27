import numpy as np
import random

class ProbabilityRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.probability_refinement = 0.15
        self.population_memory = []

    def initialize_population(self):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        return population

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the population
            evaluations = func(self.population)

            # Store the best solution
            best_idx = np.argmin(evaluations)
            best_solution = self.population[best_idx]
            self.population_memory.append(best_solution)

            # Select parents
            parents = self.select_parents(evaluations)

            # Generate offspring
            offspring = self.generate_offspring(parents, evaluations)

            # Apply mutation and crossover
            self.population = self.apply_mutation_and_crossover(self.population, offspring, self.mutation_rate, self.crossover_rate)

            # Refine the strategy using probability refinement
            self.refine_strategy(self.population, self.probability_refinement)

        # Return the best solution found
        return self.population[np.argmin(evaluations)]

    def select_parents(self, evaluations):
        # Select parents based on the fitness
        parents = []
        for _ in range(self.dim):
            idx = np.random.choice(len(evaluations))
            parents.append(self.population[idx])
        return np.array(parents)

    def generate_offspring(self, parents, evaluations):
        # Generate offspring using crossover and mutation
        offspring = []
        for _ in range(self.dim):
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            offspring.append(child)
        return np.array(offspring)

    def crossover(self, parent1, parent2):
        # Perform crossover using the crossover rate
        if random.random() < self.crossover_rate:
            child = (parent1 + parent2) / 2
            return child
        else:
            return parent1

    def mutate(self, solution):
        # Perform mutation using the mutation rate
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.dim - 1)
            solution[idx] += random.uniform(-1.0, 1.0)
            return solution
        else:
            return solution

    def apply_mutation_and_crossover(self, population, offspring, mutation_rate, crossover_rate):
        # Apply mutation and crossover to the population
        new_population = population.copy()
        for i in range(len(population)):
            if random.random() < mutation_rate:
                new_population[i] = self.mutate(population[i])
            else:
                new_population[i] = self.crossover(population[i], population[i])
        return np.vstack((new_population, offspring))

    def refine_strategy(self, population, probability_refinement):
        # Refine the strategy using probability refinement
        for i in range(self.budget):
            # Change individual lines with probability 0.15
            if random.random() < probability_refinement:
                idx = random.randint(0, self.dim - 1)
                population[i, idx] += random.uniform(-1.0, 1.0)

# Example usage
import numpy as np
from scipy.optimize import minimize

def func(x):
    return x**2 + 2*x + 1

budget = 100
dim = 10
prob_refinement = ProbabilityRefinement(budget, dim)
result = prob_refinement(func)
print(result)