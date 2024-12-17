import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.successful_deltas = []
        self.successful_crs = []

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        evals = self.population_size

        while evals < self.budget:
            new_population = np.copy(population)

            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.successful_deltas.append(np.linalg.norm(trial - population[i]))
                    self.successful_crs.append(self.CR)

                if len(self.successful_deltas) > self.population_size:
                    self.update_parameters()

            population = new_population
        
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def update_parameters(self):
        self.F = np.mean(self.successful_deltas) * 0.5 * (1 + np.random.rand() * 0.1)  # Slight modification
        self.CR = np.mean(self.successful_crs)
        self.successful_deltas = []
        self.successful_crs = []