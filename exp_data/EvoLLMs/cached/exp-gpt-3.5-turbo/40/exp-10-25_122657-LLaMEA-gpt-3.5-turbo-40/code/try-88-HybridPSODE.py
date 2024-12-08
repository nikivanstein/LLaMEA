import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 20
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.F = 0.5

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))

        def evaluate_population(population):
            return np.array([func(solution) for solution in population])

        def update_population(population, fitness):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                vel1 = self.w * population[i] + self.c1 * r1 * (population[np.random.randint(self.num_particles)] - population[i])
                vel2 = self.w * population[i] + self.c2 * r2 * (population[np.random.randint(self.num_particles)] - population[i])
                u = population[i] + self.F * (vel1 - vel2)
                new_solution = np.clip(u, self.lower_bound, self.upper_bound)
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

hybrid_pso_de = HybridPSODE(budget=1000, dim=10)  # Example usage