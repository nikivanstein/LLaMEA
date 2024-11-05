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
        stagnation_threshold = 0.01
        best_fitness = np.min(fitness)
        stagnation_count = 0

        while eval_count < self.budget:
            for i in range(pop_size):
                # Mutation with adaptive factor
                F = 0.5 + 0.3 * np.random.rand()
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

                # Early stopping criterion
                current_best_fitness = np.min(fitness)
                if abs(best_fitness - current_best_fitness) < stagnation_threshold:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                best_fitness = current_best_fitness

                if stagnation_count > 5:  # Early stopping if no improvement
                    return population[np.argmin(fitness)]
                
                if eval_count >= self.budget:
                    break
        
        best_index = np.argmin(fitness)
        return population[best_index]