import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.crossover_rate = 0.9
        self.scale_factor = 0.8
        self.max_velocity = 0.2
        self.w = 0.5
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def clip(x):
            return np.clip(x, self.lb, self.ub)

        def de_mutate(population, target_idx):
            candidates = [idx for idx in range(self.population_size) if idx != target_idx]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            return clip(population[a] + self.scale_factor * (population[b] - population[c]))

        def pso_update_position(position, velocity):
            return clip(position + velocity)

        def pso_update_velocity(velocity, particle_position, global_best_position):
            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)
            cognitive = self.c1 * r1 * (particle_position - position)
            social = self.c2 * r2 * (global_best_position - position)
            return np.clip(self.w * velocity + cognitive + social, -self.max_velocity, self.max_velocity)

        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness_values = [func(individual) for individual in population]
        global_best_position = population[np.argmin(fitness_values)]
        global_best_fitness = np.min(fitness_values)

        for _ in range(self.budget - self.population_size):
            new_population = []
            new_fitness_values = []
            for i in range(self.population_size):
                mutated = de_mutate(population, i)
                velocity = pso_update_velocity(np.zeros(self.dim), population[i], global_best_position)
                position = pso_update_position(population[i], velocity)
                new_fitness = func(position)
                if new_fitness < fitness_values[i]:
                    population[i] = position
                    fitness_values[i] = new_fitness
                if new_fitness < global_best_fitness:
                    global_best_position = position
                    global_best_fitness = new_fitness

        return global_best_position