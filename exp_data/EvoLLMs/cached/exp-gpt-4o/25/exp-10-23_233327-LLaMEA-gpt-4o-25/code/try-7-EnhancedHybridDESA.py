import numpy as np

class EnhancedHybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.adaptive_factor = 0.1  # Adaptive component for mutation

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness)]

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                indices = np.random.choice([idx for idx in range(self.pop_size) if idx != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                success_rate = (self.budget - evaluations) / self.budget  # Dynamic success rate
                
                # Adaptive mutation factor based on success rate
                adaptive_mutation = self.mutation_factor + self.adaptive_factor * success_rate
                mutant_vector = np.clip(x1 + adaptive_mutation * (x2 - x3), self.lower_bound, self.upper_bound)
                
                # Self-adaptive crossover strategy
                adaptive_crossover_rate = self.crossover_rate * (1 - success_rate)
                crossover_mask = np.random.rand(self.dim) < adaptive_crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial_vector
                else:
                    prob_accept = np.exp((fitness[i] - trial_fitness) / self.temperature)
                    if np.random.rand() < prob_accept:
                        population[i] = trial_vector
                        fitness[i] = trial_fitness

            self.temperature *= self.cooling_rate

        return best_individual, best_fitness