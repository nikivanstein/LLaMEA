import numpy as np

class PSO_DE_Hybrid:
    def __init__(self, budget, dim, population_size=30, c1=2.0, c2=2.0, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.c1 = c1
        self.c2 = c2
        self.f = f
        self.cr = cr

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def pso_de_step(population, velocities, best_individual):
            new_population = []
            for idx, (target, velocity) in enumerate(zip(population, velocities)):
                p_best = population[np.argmin([func(ind) for ind in population])]
                g_best = best_individual
                new_velocity = self.c1 * np.random.rand(self.dim) * (p_best - target) + self.c2 * np.random.rand(self.dim) * (g_best - target)
                new_velocity = np.clip(new_velocity, -1, 1)
                new_position = target + new_velocity
                if np.random.rand() < self.cr:
                    de_mutant = target + self.f * (population[np.random.randint(0, self.population_size)] - target)
                    crossover_points = np.random.rand(self.dim) < 0.5
                    new_position = np.where(crossover_points, de_mutant, new_position)
                new_population.append(new_position)
            return np.array(new_population), np.array(new_velocity)

        population = initialize_population()
        velocities = np.random.rand(self.population_size, self.dim)
        best_individual = population[np.argmin([func(ind) for ind in population])]
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            new_population, velocities = pso_de_step(population, velocities, best_individual)
            for idx, individual in enumerate(new_population):
                if remaining_budget <= 0:
                    break
                new_fitness = func(individual)
                if new_fitness < func(population[idx]):
                    population[idx] = individual
                    if new_fitness < func(best_individual):
                        best_individual = individual
                remaining_budget -= 1

        return best_individual

# Example usage:
# optimizer = PSO_DE_Hybrid(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function