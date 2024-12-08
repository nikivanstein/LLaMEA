import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.mutation_f = 0.5
        self.crossover_prob = 0.7
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim)
        )
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0
        self.best_fitness = np.inf
        self.dynamic_dim = dim

    def __call__(self, func):
        self.evaluate_population(func)
        
        while self.used_budget < self.budget:
            for i in range(self.population_size):
                if self.used_budget >= self.budget:
                    break

                # Mutation strategy with best individual influence
                best_idx = np.argmin(self.fitness)
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_f * (b - c) + 0.5 * (self.population[best_idx] - a), 
                                 self.lower_bound, self.upper_bound)
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

            # Adaptive mechanisms based on improvement
            if self.used_budget > self.budget * 0.5 and self.population_size > self.dim:
                self.population_size = int(self.initial_population_size / 2)

            # Adaptive mutation factor and crossover probability based on improvement
            current_best = np.min(self.fitness)
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.mutation_f = 0.4 + 0.3 * np.random.rand()
                self.crossover_prob = 0.6 + 0.3 * np.random.rand()
                
            # Periodic dimensional reduction to focus local search
            if self.used_budget % (self.budget // 10) == 0 and self.dynamic_dim > 2:
                self.dynamic_dim = max(2, self.dynamic_dim - 1)
                self.population = self.population[:, :self.dynamic_dim]
                self.fitness = self.fitness[:self.dynamic_dim]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.used_budget >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.used_budget += 1