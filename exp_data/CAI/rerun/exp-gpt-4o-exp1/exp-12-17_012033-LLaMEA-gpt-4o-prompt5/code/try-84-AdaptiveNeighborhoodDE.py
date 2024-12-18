import numpy as np

class AdaptiveNeighborhoodDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Increased initial population size
        self.scaling_factor = 0.6  # Initial scaling factor
        self.crossover_rate = 0.8  # Adjust crossover rate
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.success_rate = 0.1
        self.learning_factor = 0.05  # Introduce a learning factor

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
                
                mutant = np.clip(a + self.scaling_factor * (b - c), self.lower_bound, self.upper_bound)  # Use a different mutation strategy

                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < self.crossover_rate  # Vectorized crossover
                trial[crossover_points] = mutant[crossover_points]

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.success_rate = (1.0 - self.learning_factor) * self.success_rate + self.learning_factor

                if eval_count % (self.population_size * 2) == 0:
                    diversity = np.std(self.population, axis=0).mean()
                    self.scaling_factor = np.clip(0.4 + 0.3 * self.success_rate + 0.2 * diversity, 0.3, 0.9)
                    self.crossover_rate = np.clip(0.75 + 0.15 * self.success_rate + 0.1 * diversity, 0.75, 0.95)  # Adjust crossover rate limits

            if eval_count % (self.population_size * 3) == 0 and self.population_size > 5 * self.dim:
                self.population_size = max(5 * self.dim, self.population_size - int(5 * (1 + np.random.rand() * 0.1)))

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]