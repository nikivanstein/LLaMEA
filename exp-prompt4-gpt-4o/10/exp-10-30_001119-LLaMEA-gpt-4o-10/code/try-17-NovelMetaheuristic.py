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

        while evals < self.budget:
            # Adaptive population size based on remaining budget
            adaptive_population_size = max(5, int(self.population_size * (1 - evals / self.budget)))

            for i in range(adaptive_population_size):
                # Differential Evolution Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                # Adaptive mutation factor
                F_adaptive = F * (1 - evals / self.budget)
                mutant = np.clip(x1 + F_adaptive * (x2 - x3), self.lower_bound, self.upper_bound)

                # Crossover with adaptive rate
                CR_adaptive = CR * (0.9 + 0.1 * np.random.rand())
                cross_points = np.random.rand(self.dim) < CR_adaptive
                trial = np.where(cross_points, mutant, self.population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                # Simulated Annealing Acceptance with adaptive cooling
                acceptance_prob = np.exp((self.best_fitness - trial_fitness) / max(1e-9, temperature))
                if trial_fitness < self.best_fitness or np.random.rand() < acceptance_prob:
                    self.population[i] = trial
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial

            # Elitism: Retain the best solution found with a more robust mechanism
            if self.best_solution is not None:
                replace_index = np.random.randint(self.population_size // 2)
                self.population[replace_index] = self.best_solution

            # Adaptive cooling schedule
            temperature *= (0.99 + 0.01 * evals / self.budget)

        return self.best_solution