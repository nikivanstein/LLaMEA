import numpy as np

class HybridCuckooDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iter = budget // self.population_size
        self.pa = 0.25
        self.cuckoo_params = {'pa': 0.25, 'beta': 1.5}
        self.de_params = {'f': 0.5, 'cr': 0.9}

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def levy_flight(size):
            sigma_u = (np.math.gamma(1 + self.cuckoo_params['beta']) * np.sin(np.pi * self.cuckoo_params['beta'] / 2) / np.math.gamma((1 + self.cuckoo_params['beta']) / 2) * (1 ** (self.cuckoo_params['beta'] - 1))) ** (1 / self.cuckoo_params['beta'])
            sigma_v = 1
            u = np.random.normal(0, sigma_u, size)
            v = np.random.normal(0, sigma_v, size)
            return u / (np.absolute(v) ** (1 / self.cuckoo_params['beta']))

        def cuckoo_search_move(x, best):
            new_x = x + levy_flight(self.dim) * (x - best) * self.cuckoo_params['pa']
            new_x = np.clip(new_x, -5.0, 5.0)
            return new_x

        def differential_evolution_move(population, idx, best):
            r1, r2, r3 = np.random.choice(np.delete(np.arange(self.population_size), idx), 3, replace=False)
            mutant_vector = population[r1] + self.de_params['f'] * (population[r2] - population[r3])
            crossover_points = np.random.rand(self.dim) < self.de_params['cr']
            trial_vector = np.where(crossover_points, mutant_vector, population[idx])
            trial_vector = np.clip(trial_vector, -5.0, 5.0)
            return trial_vector

        population = initialize_population()
        fitness_values = np.array([objective_function(ind) for ind in population])
        best_idx = np.argmin(fitness_values)
        best = population[best_idx].copy()

        for _ in range(self.max_iter):
            for idx, ind in enumerate(population):
                if np.random.rand() < self.pa:
                    population[idx] = cuckoo_search_move(ind, best)
                else:
                    population[idx] = differential_evolution_move(population, idx, best)

            new_fitness_values = np.array([objective_function(ind) for ind in population])
            best_idx = np.argmin(new_fitness_values)

            if new_fitness_values[best_idx] < fitness_values[best_idx]:
                best = population[best_idx]

            fitness_values = new_fitness_values

        return best