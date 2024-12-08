import numpy as np
import random

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func, bounds):
        def objective(x):
            return func(x)

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual, individual] + 1e-6:
                self.fitnesses[individual, individual] = fitness
                return individual
            else:
                return individual

        def bounds_check(individual):
            return np.all(bounds(individual) <= individual)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                if not bounds_check(x):
                    x = random.uniform(bounds[x].min(), bounds[x].max())
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        # Select the individual with the best fitness
        selected_individual = np.argmax(self.fitnesses)

        # Refine the solution using a novel strategy
        refined_individual = self.refine_solution(selected_individual)

        return refined_individual

    def refine_solution(self, selected_individual):
        # Define a novel strategy to refine the solution
        # This strategy combines exploration and exploitation strategies
        exploration_rate = 0.5
        exploitation_rate = 0.5

        # Explore the search space
        for _ in range(int(self.budget * exploration_rate)):
            new_individual = random.uniform(-5.0, 5.0, self.dim)
            if not np.any(self.fitnesses[new_individual, :] < self.fitnesses[selected_individual, :]):
                new_individual = self.evaluate_fitness(new_individual)

        # Exploit the best individual found so far
        for _ in range(int(self.budget * exploitation_rate)):
            if not np.any(self.fitnesses[new_individual, :] < self.fitnesses[selected_individual, :]):
                new_individual = self.evaluate_fitness(new_individual)
            else:
                break

        return new_individual

# Test the algorithm
func = lambda x: x**2
bounds = lambda x: (x.min(), x.max())
algo = NovelMetaheuristic(100, 2)
selected_individual = algo(func, bounds)
print(selected_individual)

# Evaluate the fitness of the selected individual
fitness = algo(func, bounds)
print(fitness)