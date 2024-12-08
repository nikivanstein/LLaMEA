import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Population size
        pop_size = min(100, self.budget // self.dim)
        # Generate initial population
        population = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(pop_size, self.dim)
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size  # Initial evaluations

        # DE control parameters
        F = 0.5  # mutation factor
        CR = 0.9  # crossover probability

        while eval_count < self.budget:
            for i in range(pop_size):
                # Mutation
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover_mask, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Adaptive scaling adjustment
                if eval_count % (pop_size // 2) == 0:
                    F = 0.4 + 0.1 * (1 - eval_count / self.budget)
                
                if eval_count >= self.budget:
                    break
        
        best_index = np.argmin(fitness)
        return population[best_index]