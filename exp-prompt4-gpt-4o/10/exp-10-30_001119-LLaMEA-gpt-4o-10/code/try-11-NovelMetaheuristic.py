import numpy as np

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        evals = 0
        temperature = 1.0
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        prev_best_fitness = self.best_fitness

        while evals < self.budget:
            adaptive_population_size = max(5, int(self.population_size * (1 - evals / self.budget)))

            for i in range(adaptive_population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                F_adaptive = F * (1 - evals / self.budget)
                mutant = np.clip(x1 + F_adaptive * (x2 - x3), self.lower_bound, self.upper_bound)

                CR_adaptive = CR * (0.9 + 0.1 * np.random.rand())
                cross_points = np.random.rand(self.dim) < CR_adaptive
                trial = np.where(cross_points, mutant, self.population[i])

                levy_flight = 0.01 * np.random.randn(self.dim) * (np.divide(1, np.power(np.abs(np.random.randn(self.dim)), 1.5)))
                trial = np.clip(trial + levy_flight, self.lower_bound, self.upper_bound)

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < self.best_fitness or np.random.rand() < np.exp((self.best_fitness - trial_fitness) / temperature):
                    self.population[i] = trial
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial
            
            if np.abs(prev_best_fitness - self.best_fitness) < 1e-8:
                break
            prev_best_fitness = self.best_fitness

            self.population[np.random.randint(self.population_size)] = self.best_solution
            temperature *= 0.99

        return self.best_solution