import numpy as np

class E_HAFA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iter = budget // self.population_size

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def firefly_move(x, attractiveness, best, alpha=0.2):
            r = np.random.uniform(-1, 1, size=self.dim)
            distance = np.linalg.norm(x - best)
            x += attractiveness * np.exp(-alpha * distance) * r
            x = np.clip(x, -5.0, 5.0)
            return x

        def de_move(x, population, f=0.5, cr=0.7):
            a, b, c = np.random.choice(population, 3, replace=False)
            mutant = np.clip(a + f * (b - c), -5.0, 5.0)
            crossover = np.random.rand(self.dim) < cr
            trial = np.where(crossover, mutant, x)
            return trial

        population = initialize_population()
        fitness_values = np.array([objective_function(ind) for ind in population])
        best_idx = np.argmin(fitness_values)
        best = population[best_idx].copy()

        for _ in range(self.max_iter):
            for idx, ind in enumerate(population):
                attractiveness = 1 / (1 + fitness_values[idx])
                if np.random.rand() < 0.5:  # 0.25 probability to use DE strategy
                    population[idx] = de_move(ind, population)
                else:
                    population[idx] = firefly_move(ind, attractiveness, best)

            new_fitness_values = np.array([objective_function(ind) for ind in population])
            best_idx = np.argmin(new_fitness_values)

            if new_fitness_values[best_idx] < fitness_values[best_idx]:
                best = population[best_idx]

            fitness_values = new_fitness_values

        return best