import numpy as np
import random

class PAHEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.population = []
        self.fitness = []
        self.best_solution = None
        self.best_score = -np.inf

    def __call__(self, func):
        # Initialize population with random solutions
        for _ in range(self.population_size):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            self.population.append(solution)
            self.fitness.append(func(solution))

        # Evaluate the fitness of each solution
        for i in range(self.population_size):
            self.fitness[i] = -self.fitness[i]  # Use negative fitness for maximization problems

        # Select parents using tournament selection
        parents = []
        for _ in range(self.population_size):
            tournament_size = 5
            tournament = random.sample(range(self.population_size), tournament_size)
            winner = max(tournament, key=lambda i: self.fitness[i])
            parents.append(self.population[winner])

        # Apply probability-adaptive mutation
        for _ in range(self.population_size):
            if random.random() < 0.3:
                mutation_rate = random.random()
                if mutation_rate < 0.5:
                    # Swap two random components of the solution
                    i, j = random.sample(range(self.dim), 2)
                    parents[_][i], parents[_][j] = parents[_][j], parents[_][i]

        # Apply hybrid evolution
        for _ in range(self.budget - self.population_size):
            # Select parents using tournament selection
            parents = []
            for _ in range(self.population_size):
                tournament_size = 5
                tournament = random.sample(range(self.population_size), tournament_size)
                winner = max(tournament, key=lambda i: self.fitness[i])
                parents.append(self.population[winner])

            # Apply crossover
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                crossover_point = random.randint(1, self.dim - 1)
                child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                offspring.append(child)

            # Apply mutation
            for _ in range(self.population_size):
                if random.random() < 0.3:
                    mutation_rate = random.random()
                    if mutation_rate < 0.5:
                        # Swap two random components of the solution
                        i, j = random.sample(range(self.dim), 2)
                        offspring[_][i], offspring[_][j] = offspring[_][j], offspring[_][i]

            # Replace the worst solution with the new offspring
            worst_solution = min(offspring, key=lambda x: self.fitness[self.fitness.index(min(self.fitness))])
            self.population[self.fitness.index(worst_solution)] = worst_solution

        # Update the best solution
        self.best_solution = max(self.population, key=lambda x: self.fitness[self.fitness.index(max(self.fitness))])
        self.best_score = -self.fitness[self.fitness.index(max(self.fitness))]

# Example usage:
def func(x):
    return np.sum(x**2)

pahea = PAHEA(100, 10)
pahea(func)