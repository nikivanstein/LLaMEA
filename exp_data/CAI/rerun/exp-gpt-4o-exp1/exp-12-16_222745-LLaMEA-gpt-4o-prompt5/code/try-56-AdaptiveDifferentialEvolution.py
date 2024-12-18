import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5
        self.CR = 0.9 * np.random.uniform(0.8, 1.2)
        self.successful_deltas = []
        self.successful_crs = []
        self.adaptive_scaling = True
        self.adaptive_crossover = True  # Change 1: Enable adaptive crossover

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
                F_dynamic = self.F * np.random.uniform(0.4, 1.6) if self.adaptive_scaling else self.F 
                mutant = np.clip(a + F_dynamic * (b - c), self.lower_bound, self.upper_bound)
                
                if self.adaptive_crossover:  # Change 2: Adjust crossover rate dynamically
                    dynamic_CR = min(max(0.1, self.CR + 0.1 * (np.median(self.successful_crs) - self.CR)), 0.9)
                else:
                    dynamic_CR = self.CR * np.random.uniform(0.8, 1.2)
                
                cross_points = np.random.rand(self.dim) < dynamic_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.successful_deltas.append(np.linalg.norm(trial - population[i]))
                    self.successful_crs.append(dynamic_CR)

            if len(self.successful_deltas) > self.population_size:
                self.update_parameters()

            improvement_rate = np.sum(np.array(self.successful_deltas) < np.mean(self.successful_deltas)) / len(self.successful_deltas)
            self.population_size = int(max(4, improvement_rate * 10 * self.dim))

            population = new_population
        
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def update_parameters(self):
        self.F = 0.5 + 0.3 * (np.random.rand() - 0.5) + 0.1 * (1 - np.exp(-np.std(self.successful_deltas)))
        self.CR = np.mean(self.successful_crs) + 0.1 * (1 - np.exp(-np.std(self.successful_crs)))
        self.successful_deltas = []
        self.successful_crs = []