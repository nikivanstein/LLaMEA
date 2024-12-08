import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 12 * dim  # Increased initial population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.fitness = np.full(self.initial_population_size, np.inf)
        self.F = 0.9  # Adjusted differential weight
        self.CR = 0.8  # Adjusted initial crossover probability
        self.evaluations = 0

    def __call__(self, func):
        population_size = self.initial_population_size
        for i in range(population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1

        while self.evaluations < self.budget:
            fitness_std = np.std(self.fitness)  # Calculate fitness diversity
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break

                # Mutation
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                # Adaptive mutation scaling
                adaptive_F = 0.5 + (0.5 * fitness_std / (np.abs(np.mean(self.fitness)) + 1e-8))
                self.F = np.clip(adaptive_F, 0.4, 1.0)

                # Adaptive crossover probability
                self.CR = 0.8 - (0.3 * (self.evaluations / self.budget))
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.evaluations += 1

                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.population[i] = trial

            # Dynamic population control
            if self.evaluations > self.budget * 0.6 and population_size > 5:
                population_size = max(5, int(population_size * 0.9))
                self.population = self.population[:population_size]
                self.fitness = self.fitness[:population_size]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]