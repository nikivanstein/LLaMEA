import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Slightly increased population size
        self.mutation_f = 0.5
        self.crossover_prob = 0.7
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim)
        )
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0

    def __call__(self, func):
        self.evaluate_population(func)
        
        while self.used_budget < self.budget:
            for i in range(self.population_size):
                if self.used_budget >= self.budget:
                    break

                # Mutation and recombination
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_f * (b - c), self.lower_bound, self.upper_bound)
                trial = np.copy(self.population[i])
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial[cross_points] = mutant[cross_points]

                # Selection
                trial_fitness = func(trial)
                self.used_budget += 1
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial

            # Adaptation of mutation factor and crossover probability
            self.mutation_f = 0.5 + 0.3 * (np.mean(self.fitness) - np.min(self.fitness)) / (np.max(self.fitness) - np.min(self.fitness) + 1e-9)
            self.crossover_prob = 0.5 + 0.4 * np.random.rand()

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.used_budget >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.used_budget += 1