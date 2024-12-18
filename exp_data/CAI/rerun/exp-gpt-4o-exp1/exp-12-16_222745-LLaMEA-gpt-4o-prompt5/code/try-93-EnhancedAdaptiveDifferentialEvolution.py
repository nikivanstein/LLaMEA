import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50  # Change: Fixed initial population size
        self.F = 0.5
        self.CR = 0.9
        self.successful_deltas = []
        self.successful_crs = []
        self.adaptive_scaling = True

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
                F_dynamic = self.F * np.random.uniform(0.5, 2.0)  # Change: Wider F_dynamic range for exploration
                mutant = np.clip(a + F_dynamic * (b - c), self.lower_bound, self.upper_bound)
                
                dynamic_CR = self.CR * np.random.uniform(0.6, 1.4)  # Change: Adjusted range for dynamic_CR
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

            decrease_rate = np.exp(-np.var(fitness))  # Change: New dynamic population adjustment
            self.population_size = int(min(max(4, self.population_size * decrease_rate), 100))  # Change: Adaptive population size limit

            population = new_population
        
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def update_parameters(self):
        self.F = 0.5 + 0.2 * (np.log1p(np.random.rand()) - 0.5) + 0.15 * (1 - np.exp(-np.std(self.successful_deltas)))  # Change: Fine-tuned log-scale adjustment
        self.CR = np.mean(self.successful_crs) + 0.15 * (1 - np.exp(-np.std(self.successful_crs)))  # Change: Increased adjustment
        self.successful_deltas = []
        self.successful_crs = []