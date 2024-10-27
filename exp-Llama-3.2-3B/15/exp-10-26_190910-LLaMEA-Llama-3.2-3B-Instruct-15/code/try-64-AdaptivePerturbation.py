import numpy as np
import random

class AdaptivePerturbation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()
        self.perturbation_rate = 0.15
        self.perturbation_strategies = {
            'uniform': lambda x: x + np.random.uniform(-1.0, 1.0),
            'gaussian': lambda x: x + np.random.normal(0.0, 1.0),
            'random': lambda x: x + np.random.choice([-1.0, 1.0])
        }

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

            # Apply perturbation
            self.population = self.apply_perturbation(self.population, self.perturbation_strategies, self.perturbation_rate)

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
        # Generate offspring using crossover and perturbation
        offspring = []
        for _ in range(self.dim):
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            child = self.perturb(child)
            offspring.append(child)
        return np.array(offspring)

    def crossover(self, parent1, parent2):
        # Perform crossover using the crossover rate
        if random.random() < 0.7:
            child = (parent1 + parent2) / 2
            return child
        else:
            return parent1

    def perturb(self, solution):
        # Apply perturbation with probability 0.15
        if random.random() < self.perturbation_rate:
            strategy = np.random.choice(list(self.perturbation_strategies.keys()))
            return self.perturbation_strategies[strategy](solution)
        else:
            return solution

    def apply_perturbation(self, population, perturbation_strategies, perturbation_rate):
        # Apply perturbation to the population
        new_population = population.copy()
        for i in range(len(population)):
            if random.random() < perturbation_rate:
                strategy = np.random.choice(list(perturbation_strategies.keys()))
                new_population[i] = perturbation_strategies[strategy](new_population[i])
        return new_population