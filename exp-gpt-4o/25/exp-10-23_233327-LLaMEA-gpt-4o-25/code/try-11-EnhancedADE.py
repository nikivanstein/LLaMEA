import numpy as np

class EnhancedADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 25  # Increased population for diversity
        self.mutation_factor = 0.7 + 0.3 * np.random.rand()  # Adaptive mutation factor
        self.crossover_rate = 0.85  # Slightly reduced crossover rate
        self.temperature = 1.0
        self.cooling_rate = 0.93  # Faster cooling
        self.chaos_factor = np.random.uniform(0.5, 1.5)  # Chaos-driven factor

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant_vector = np.clip(x1 + self.mutation_factor * (x2 - x3), self.lower_bound, self.upper_bound)
                mutant_vector += self.chaos_factor * (x1 - mutant_vector)  # Chaos-influenced exploration
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                else:
                    # Soft elitism strategy
                    if np.random.rand() < 0.1:
                        population[i] = trial_vector
                        fitness[i] = trial_fitness

            self.temperature *= self.cooling_rate
            self.mutation_factor = 0.5 + 0.5 * np.random.rand()  # Update mutation factor adaptively

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]