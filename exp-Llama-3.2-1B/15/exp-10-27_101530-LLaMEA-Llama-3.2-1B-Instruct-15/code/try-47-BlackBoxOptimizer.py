import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Select parents using tournament selection
            parents = self.select_parents(population, self.func_evaluations)
            # Create offspring by crossover and mutation
            offspring = self.crossover(parents, self.dim)
            # Evaluate the new offspring
            new_individual = self.mutate(offspring, self.func_evaluations)
            # Check if the new individual is within the budget
            if self.func_evaluations < self.budget:
                # If not, replace the worst individual with the new one
                self.population[self.func_evaluations] = new_individual
                self.func_evaluations += 1
            else:
                # If the budget is reached, return the best individual found so far
                return self.population[self.func_evaluations - 1]
        # If the budget is reached, return the best individual found so far
        return self.population[self.func_evaluations - 1]

    def select_parents(self, population, func_evaluations):
        # Select parents using tournament selection
        tournament_size = 5
        parents = []
        for _ in range(func_evaluations):
            individual = random.choice(population)
            winner = 0
            for p in population:
                if p!= individual and np.random.rand() < np.exp((p - individual) / 10):
                    winner = p
            parents.append(winner)
        return parents

    def crossover(self, parents, dim):
        # Create offspring by crossover
        offspring = []
        for i in range(len(parents)):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % len(parents)]
            crossover_point = random.randint(1, dim)
            offspring.append(parent1[:crossover_point] + parent2[crossover_point:])
        return offspring

    def mutate(self, offspring, func_evaluations):
        # Mutate the offspring
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = [random.uniform(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
            mutated_individual[0] += random.uniform(-5, 5)
            mutated_individual[1] += random.uniform(-5, 5)
            mutated_individual = tuple(mutated_individual)
            if np.random.rand() < 0.1:
                mutated_individual = tuple(random.uniform(self.search_space[0], self.search_space[1]) for _ in range(self.dim))
            mutated_offspring.append(mutated_individual)
        return mutated_offspring

    def evaluate_fitness(self, individual, func):
        # Evaluate the fitness of the individual
        return func(individual)

# Example usage
def black_box_optimization():
    optimizer = BlackBoxOptimizer(100, 10)
    func = lambda x: x**2
    best_individual = optimizer(100)
    print("Best individual:", best_individual)
    print("Best fitness:", optimizer.evaluate_fitness(best_individual, func))

black_box_optimization()