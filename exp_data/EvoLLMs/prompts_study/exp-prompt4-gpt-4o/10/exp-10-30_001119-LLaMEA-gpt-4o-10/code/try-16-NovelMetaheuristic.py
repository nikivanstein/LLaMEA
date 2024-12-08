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
                
                # Dynamic mutation factor based on diversity
                diversity = np.std(self.population, axis=0).mean()
                F_adaptive = F * (1 + diversity)
                
                mutant = np.clip(x1 + F_adaptive * (x2 - x3), self.lower_bound, self.upper_bound)
                
                # Dynamic crossover rate based on progress
                progress = evals / self.budget
                CR_adaptive = CR * (0.8 + 0.2 * np.sin(progress * np.pi))
                cross_points = np.random.rand(self.dim) < CR_adaptive
                trial = np.where(cross_points, mutant, self.population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                # Simulated Annealing Acceptance
                if trial_fitness < self.best_fitness or np.random.rand() < np.exp((self.best_fitness - trial_fitness) / temperature):
                    self.population[i] = trial
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial

            # Elitism: Retain the best solution found
            self.population[np.random.randint(self.population_size)] = self.best_solution

            # Cooling schedule
            temperature *= 0.99

        return self.best_solution