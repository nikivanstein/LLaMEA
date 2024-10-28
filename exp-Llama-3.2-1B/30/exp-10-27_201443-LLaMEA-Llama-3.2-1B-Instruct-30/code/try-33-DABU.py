import numpy as np
import random

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def __str__(self):
        return f"Population: {self.func_evaluations}/{self.budget}"

class BBOOpt:
    def __init__(self, algorithm, budget, dim):
        self.algorithm = algorithm
        self.budget = budget
        self.dim = dim

    def optimize(self, func, initial_solution, iterations=1000):
        # Initialize the population with random solutions
        population = [initial_solution]
        for _ in range(iterations):
            # Evaluate the fitness of each solution
            fitnesses = [self.algorithm(func, solution) for solution in population]
            # Select the fittest solutions
            self.algorithm.select(population, fitnesses)
            # Generate new solutions by perturbing the fittest solutions
            new_population = []
            for _ in range(self.budget):
                new_solution = [random.uniform(self.algorithm.search_space[i] - 1, self.algorithm.search_space[i] + 1) for i in range(self.dim)]
                # Ensure the new solution is within the search space
                new_solution = [max(self.algorithm.search_space[i], min(new_solution[i], self.algorithm.search_space[i])) for i in range(self.dim)]
                new_population.append(new_solution)
            population = new_population
        return population

    def select(self, population, fitnesses):
        # Use tournament selection to select the fittest solutions
        tournament_size = 3
        winners = []
        for _ in range(len(population)):
            winner_index = random.randint(0, len(population) - 1)
            winner = population[winner_index]
            for _ in range(tournament_size):
                winner_index = random.randint(0, len(population) - 1)
                winner = population[winner_index]
                if fitnesses[winner_index] > fitnesses[winner]:
                    winner = winner
                    break
            winners.append(winner)
        return winners

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

def bbo_optimize(func, initial_solution, budget=1000, dim=2):
    algorithm = DABU(budget, dim)
    population = [initial_solution]
    for _ in range(budget):
        population = algorithm.optimize(func, population)
    return population

# Run the BBO optimization algorithm
population = bbo_optimize(test_function, [0, 0], dim=2)
print(f"Population: {len(population)}")
print(f"Fitness: {min(fitnesses) / len(fitnesses)}")