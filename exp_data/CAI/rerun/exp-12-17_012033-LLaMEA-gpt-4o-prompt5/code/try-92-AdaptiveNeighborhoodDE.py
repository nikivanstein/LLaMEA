import numpy as np

class AdaptiveNeighborhoodDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.scaling_factor = 0.5
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.success_rate = 0.1

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            best_idx = np.argmin(self.fitness)
            best_individual = self.population[best_idx]
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + self.scaling_factor * (b - c), self.lower_bound, self.upper_bound)

                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < self.crossover_rate
                trial[crossover_points] = mutant[crossover_points]

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.success_rate = 0.85 * self.success_rate + 0.15

            if eval_count % (self.population_size * 2) == 0:
                diversity = np.std(self.population, axis=0).mean()
                self.scaling_factor = np.clip(0.3 + 0.5 * self.success_rate + 0.2 * diversity, 0.3, 0.9)
                self.crossover_rate = np.clip(0.8 + 0.1 * self.success_rate + 0.1 * diversity, 0.8, 1.0)
            
            if eval_count % (self.population_size * 5) == 0:
                if eval_count > 0.5 * self.budget:
                    self.population_size = max(5 * self.dim, self.population_size - int(5 * (1 + 0.1 * np.random.rand())))

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]