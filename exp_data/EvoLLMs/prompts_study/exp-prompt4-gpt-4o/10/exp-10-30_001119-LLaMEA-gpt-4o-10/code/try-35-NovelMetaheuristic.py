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
        initial_CR = 0.9  # Initial crossover rate

        while evals < self.budget:
            adaptive_population_size = max(5, int(self.population_size * (1 - evals / self.budget)))

            for i in range(adaptive_population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                F_adaptive = F * (1 - evals / self.budget)
                mutant = np.clip(x1 + F_adaptive * (x2 - x3), self.lower_bound, self.upper_bound)

                diversity_factor = np.std(self.population, axis=0).mean()
                CR_adaptive = initial_CR * (0.8 + 0.2 * np.random.rand()) * (1 + diversity_factor)
                
                cross_points = np.random.rand(self.dim) < CR_adaptive
                trial = np.where(cross_points, mutant, self.population[i])

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < self.best_fitness or np.random.rand() < np.exp((self.best_fitness - trial_fitness) / temperature):
                    self.population[i] = trial
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial

            self.population[np.random.randint(self.population_size)] = self.best_solution
            temperature *= 0.99

        return self.best_solution