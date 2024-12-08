import numpy as np
import random

class AdaptiveEvolutionary:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
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

            # Update the population
            self.population = np.vstack((self.population, offspring))

            # Apply probabilistic mutation and crossover
            new_population = self.population.copy()
            for i in range(self.population.shape[0]):
                if random.random() < self.mutation_rate:
                    idx = random.randint(0, self.dim - 1)
                    new_population[i, idx] += random.uniform(-1.0, 1.0)
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.sample(parents, 2)
                    child = (parent1 + parent2) / 2
                    new_population[i] = child
            self.population = new_population

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
            child = (parent1 + parent2) / 2
            offspring.append(child)
        return np.array(offspring)

    def apply_mutation_and_crossover(self, population, mutation_rate, crossover_rate):
        # Apply mutation and crossover to the population
        new_population = population.copy()
        for i in range(population.shape[0]):
            if random.random() < mutation_rate:
                idx = random.randint(0, population.shape[1] - 1)
                new_population[i, idx] += random.uniform(-1.0, 1.0)
            if random.random() < crossover_rate:
                parent1, parent2 = random.sample(population, 2)
                child = (parent1 + parent2) / 2
                new_population[i] = child
        return new_population

# Example usage:
# bbob_functions = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23]
# for func in bbob_functions:
#     algo = AdaptiveEvolutionary(50, 10)
#     best_solution = algo(func)
#     print(func(best_solution))