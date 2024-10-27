import numpy as np
import random

class ProbabilisticAdaptiveEvolutionary:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.refine_rate = 0.15
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

            # Refine the offspring
            refined_offspring = self.refine_offspring(offspring, evaluations)

            # Update the population
            self.population = np.vstack((self.population, refined_offspring))

            # Apply mutation and crossover
            self.population = self.apply_mutation_and_crossover(self.population, self.mutation_rate, self.crossover_rate)

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

    def refine_offspring(self, offspring, evaluations):
        # Refine the offspring with a probabilistic strategy
        refined_offspring = []
        for i, individual in enumerate(offspring):
            if random.random() < self.refine_rate:
                # Refine the individual by changing a random line
                lines = individual.tolist()
                lines[i] = random.uniform(-5.0, 5.0)
                individual = np.array(lines)
            refined_offspring.append(individual)
        return np.array(refined_offspring)

    def apply_mutation_and_crossover(self, population, mutation_rate, crossover_rate):
        # Apply mutation and crossover to the population
        new_population = population.copy()
        for i in range(len(population)):
            if random.random() < mutation_rate:
                new_population[i] = self.mutate(population[i])
            else:
                new_population[i] = self.crossover(population[i], population[i])
        return new_population

    def mutate(self, solution):
        # Perform mutation using the mutation rate
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.dim - 1)
            solution[idx] += random.uniform(-1.0, 1.0)
            return solution
        else:
            return solution

    def crossover(self, parent1, parent2):
        # Perform crossover using the crossover rate
        if random.random() < self.crossover_rate:
            child = (parent1 + parent2) / 2
            return child
        else:
            return parent1