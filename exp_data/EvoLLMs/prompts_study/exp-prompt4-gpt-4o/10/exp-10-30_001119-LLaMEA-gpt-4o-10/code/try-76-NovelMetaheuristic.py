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
        self.memory = np.random.uniform(self.lower_bound, self.upper_bound, dim)

    def __call__(self, func):
        evals = 0
        temperature = 1.0
        F = 0.8
        CR = 0.9

        while evals < self.budget:
            adaptive_population_size = max(5, int(self.population_size * (1 - evals / self.budget)))
            F_adaptive_range = (0.6, 1.0)

            for i in range(adaptive_population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                F_adaptive = np.random.uniform(*F_adaptive_range)
                mutant = np.clip(x1 + F_adaptive * (x2 - x3), self.lower_bound, self.upper_bound)

                CR_adaptive = CR * (0.9 + 0.1 * np.random.rand())
                CR_adaptive *= max(0.5, 1 - evals / self.budget)  # Adaptive crossover
                cross_points = np.random.rand(self.dim) < CR_adaptive
                trial = np.where(cross_points, mutant, self.population[i])

                guided_mutation = 0.5 * (trial + self.memory)
                trial = np.clip(guided_mutation, self.lower_bound, self.upper_bound)

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < self.best_fitness or np.random.rand() < np.exp((self.best_fitness - trial_fitness) / temperature):
                    self.population[i] = trial
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial
                        self.memory = trial

            if evals % (self.budget * 0.1) == 0:
                self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
            
            self.population[np.random.randint(self.population_size)] = self.best_solution
            temperature *= 0.995  # Slower temperature decay for stability

        return self.best_solution