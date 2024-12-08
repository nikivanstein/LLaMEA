import numpy as np
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population_size = 100
        self.population = np.random.uniform(self.bounds[0][0], self.bounds[0][1], (self.population_size, self.dim))

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = self.select_parents()

            # Crossover and mutate the selected parents
            offspring = self.crossover_and_mutate(parents)

            # Evaluate the fitness of the offspring
            fitness = self.evaluate_fitness(offspring)

            # Update the population
            self.population = np.concatenate((self.population, offspring), axis=0)

            # Refine the strategy of the selected individuals
            self.refine_strategy()

        # Return the best individual
        return np.min(self.population, axis=0), np.min(self.population, axis=1)

    def select_parents(self):
        # Select 2 parents using tournament selection
        parents = []
        for _ in range(2):
            tournament = np.random.choice(self.population, 5, replace=False)
            winner = np.argmin(tournament)
            parents.append(tournament[winner])
        return parents

    def crossover_and_mutate(self, parents):
        # Perform crossover and mutation on the selected parents
        offspring = []
        for parent in parents:
            child = parent + np.random.normal(0, 0.1, self.dim)
            child = np.clip(child, self.bounds[0][0], self.bounds[0][1])
            offspring.append(child)
        return offspring

    def evaluate_fitness(self, individuals):
        # Evaluate the fitness of the individuals
        fitness = []
        for individual in individuals:
            fitness.append(func(individual))
        return fitness

    def refine_strategy(self):
        # Refine the strategy of the selected individuals
        selected_individuals = np.random.choice(self.population, 20, replace=False)
        for individual in selected_individuals:
            strategy = np.random.choice(['sbx', 'rand1'], p=[0.4, 0.6])
            if strategy =='sbx':
                individual = individual + np.random.normal(0, 0.1, self.dim)
                individual = np.clip(individual, self.bounds[0][0], self.bounds[0][1])
            elif strategy == 'rand1':
                individual = individual + np.random.normal(0, 0.1, self.dim)
                individual = np.clip(individual, self.bounds[0][0], self.bounds[0][1])
            individual = individual.tolist()

# Example usage:
func = lambda x: x[0]**2 + x[1]**2
heacombbo = HybridEvolutionaryAlgorithm(100, 2)
best_individual, best_fitness = heacombbo(func)
print(f'Best individual: {best_individual}')
print(f'Best fitness: {best_fitness}')