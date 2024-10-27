import numpy as np

class FireflyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_fireflies = 10

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.num_fireflies, self.dim))

        def evaluate_population(population):
            return np.array([func(solution) for solution in population])

        def update_population(population, fitness):
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            for i in range(self.num_fireflies):
                attractiveness = np.zeros(self.num_fireflies)
                for j in range(self.num_fireflies):
                    if fitness[j] < fitness[i]:
                        attractiveness[j] = 1 / (1 + np.linalg.norm(population[i] - population[j]))

                new_solution = population[i] + np.random.normal(0, 1, self.dim) * np.sum(attractiveness * (best_solution - population[i]))
                new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                if func(new_solution) < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = func(new_solution)

            return population, fitness

        population = initialize_population()
        fitness = evaluate_population(population)

        for _ in range(self.budget - self.budget // 10):
            population, fitness = update_population(population, fitness)

        best_idx = np.argmin(fitness)
        return population[best_idx]