import numpy as np

class AdaptiveControlDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 1.7
        self.c2 = 1.7
        self.min_inertia = 0.4
        self.max_inertia = 0.9
        self.cr = 0.8
        self.f_min = 0.4
        self.f_max = 0.9
        self.lb = -5.0 * np.ones(dim)
        self.ub = 5.0 * np.ones(dim)

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def within_bounds(x):
            return np.clip(x, self.lb, self.ub)

        def create_population():
            return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

        population = create_population()
        fitness_values = np.array([objective_function(individual) for individual in population])
        best_index = np.argmin(fitness_values)
        best_individual = population[best_index]
        gbest = best_individual.copy()
        inertia_weight = self.max_inertia

        for _ in range(self.budget - self.population_size):
            r1, r2, r3 = np.random.randint(0, self.population_size, 3)
            xr1 = population[r1]
            xr2 = population[r2]
            xr3 = population[r3]

            f = self.f_min + (_ / (self.budget - self.population_size)) * (self.f_max - self.f_min)

            mutant = within_bounds(xr1 + f * (xr2 - xr3))

            trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population[_ % self.population_size])

            v = inertia_weight * population[_ % self.population_size] + self.c1 * np.random.rand(self.dim) * (gbest - population[_ % self.population_size]) + self.c2 * np.random.rand(self.dim) * (trial - population[_ % self.population_size])

            population[_ % self.population_size] = within_bounds(v)

            fitness_values[_ % self.population_size] = objective_function(population[_ % self.population_size])

            if fitness_values[_ % self.population_size] < fitness_values[best_index]:
                best_index = _ % self.population_size
                best_individual = population[best_index]

            if fitness_values[_ % self.population_size] < objective_function(gbest):
                gbest = population[_ % self.population_size]

            inertia_weight = self.max_inertia - (_ / (self.budget - self.population_size)) * (self.max_inertia - self.min_inertia)

        return objective_function(gbest)