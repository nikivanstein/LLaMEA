import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 10
        self.num_individuals = 5
        self.c1 = 1.5
        self.c2 = 1.5

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))

        def evaluate_population(population):
            return np.array([func(solution) for solution in population])

        def update_population(population, fitness):
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            for i in range(self.num_particles):
                idxs = np.random.choice(range(self.num_particles), self.num_individuals, replace=False)
                individuals = population[idxs]

                for j in range(self.num_individuals):
                    r1 = np.random.uniform(0, 1, self.dim)
                    r2 = np.random.uniform(0, 1, self.dim)
                    new_solution = population[i] + self.c1 * r1 * (best_solution - population[i]) + self.c2 * r2 * (individuals[j] - population[i])
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