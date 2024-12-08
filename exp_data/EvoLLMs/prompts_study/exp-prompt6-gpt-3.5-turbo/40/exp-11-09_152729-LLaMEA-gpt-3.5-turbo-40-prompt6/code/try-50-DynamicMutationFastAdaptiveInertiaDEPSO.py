import numpy as np

class DynamicMutationFastAdaptiveInertiaDEPSO(FastAdaptiveInertiaDEPSO):
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
        c3 = self.max_c3
        mutation_factors = np.full(self.population_size, self.f)

        for _ in range(self.budget - self.population_size):
            r1, r2, r3 = np.random.randint(0, self.population_size, 3)
            xr1 = population[r1]
            xr2 = population[r2]
            xr3 = population[r3]

            mutant = within_bounds(xr1 + mutation_factors * (xr2 - xr3))

            trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population[_ % self.population_size])

            v = inertia_weight * population[_ % self.population_size] + self.c1 * np.random.rand(self.dim) * (gbest - population[_ % self.population_size]) + self.c2 * np.random.rand(self.dim) * (trial - population[_ % self.population_size])

            population[_ % self.population_size] = within_bounds(v)

            fitness_values[_ % self.population_size] = objective_function(population[_ % self.population_size])

            if fitness_values[_ % self.population_size] < fitness_values[best_index]:
                best_index = _ % self.population_size
                best_individual = population[best_index]

            if fitness_values[_ % self.population_size] < objective_function(gbest):
                gbest = population[_ % self.population_size]

            inertia_weight = self.max_inertia - (_ / (self.budget - self.population_size))**1.2 * (self.max_inertia - self.min_inertia)
            c3 = self.max_c3 - (_ / (self.budget - self.population_size))**1.2 * (self.max_c3 - self.min_c3)

            mutation_factors = np.clip(mutation_factors * (1 + (fitness_values - fitness_values.mean())), 0.1, 0.9)
            
            self.cr = np.clip(self.cr + np.random.normal(0.0, c3*0.9), 0.1, 0.9) 

        return objective_function(gbest)