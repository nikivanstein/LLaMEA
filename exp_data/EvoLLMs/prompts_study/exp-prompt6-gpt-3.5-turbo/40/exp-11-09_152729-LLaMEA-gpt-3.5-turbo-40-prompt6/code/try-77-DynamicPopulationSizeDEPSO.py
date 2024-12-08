# import numpy as np

class DynamicPopulationSizeDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.c1 = 1.7
        self.c2 = 1.7
        self.min_inertia = 0.4
        self.max_inertia = 0.9
        self.min_c3 = 0.1
        self.max_c3 = 0.5
        self.cr = 0.8
        self.f = 0.5
        self.lb = -5.0 * np.ones(dim)
        self.ub = 5.0 * np.ones(dim)
        self.iteration_count = 0

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def within_bounds(x):
            return np.clip(x, self.lb, self.ub)

        def create_population(pop_size):
            return np.random.uniform(self.lb, self.ub, (pop_size, self.dim))

        population_size = 30
        population = create_population(population_size)
        fitness_values = np.array([objective_function(individual) for individual in population])
        best_index = np.argmin(fitness_values)
        best_individual = population[best_index]
        gbest = best_individual.copy()
        inertia_weight = self.max_inertia
        c3 = self.max_c3

        for _ in range(self.budget - population_size):
            r1, r2, r3 = np.random.randint(0, population_size, 3)
            xr1 = population[r1]
            xr2 = population[r2]
            xr3 = population[r3]

            mutant = within_bounds(xr1 + self.f * (xr2 - xr3))

            trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population[_ % population_size])

            v = inertia_weight * population[_ % population_size] + self.c1 * np.random.rand(self.dim) * (gbest - population[_ % population_size]) + self.c2 * np.random.rand(self.dim) * (trial - population[_ % population_size])

            v = np.where(np.random.rand(self.dim) < np.clip(0.8 - 0.2 * (fitness_values[_ % population_size] - fitness_values[best_index]), 0.6, 0.8), v, gbest)

            population[_ % population_size] = within_bounds(v)

            fitness_values[_ % population_size] = objective_function(population[_ % population_size])

            if fitness_values[_ % population_size] < fitness_values[best_index]:
                best_index = _ % population_size
                best_individual = population[best_index]

            if fitness_values[_ % population_size] < objective_function(gbest):
                gbest = population[_ % population_size]

            inertia_weight = self.max_inertia - (self.iteration_count / (self.budget - population_size))**1.3 * (self.max_inertia - self.min_inertia)
            c3 = self.max_c3 - (self.iteration_count / (self.budget - population_size))**1.3 * (self.max_c3 - self.min_c3)

            self.f = np.clip(self.f + np.random.normal(0.0, c3 * 0.95), 0.1, 0.9)
            self.cr = np.clip(self.cr + np.random.normal(0.0, c3 * 0.95), 0.1, 0.9)

            self.iteration_count += 1

            # Dynamic adaptation of population size
            if self.iteration_count % 10 == 0:
                population_size = int(30 + 20 * np.sin(0.1 * self.iteration_count))
                population = create_population(population_size)
                fitness_values = np.array([objective_function(individual) for individual in population])

        return objective_function(gbest)