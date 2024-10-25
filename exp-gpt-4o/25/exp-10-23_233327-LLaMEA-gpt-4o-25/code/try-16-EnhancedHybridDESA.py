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
        self.initial_temperature = 1.0
        self.cooling_schedule = lambda t, i: t * (0.9 + 0.05 * np.random.rand())
        self.cooling_rate = 0.95
        self.diversity_control_factor = 0.1
        self.adaptive_mutation = lambda f, i: f * (0.9 + 0.2 * np.sin(np.pi * i / self.pop_size))

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        temperature = self.initial_temperature

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                adaptive_factor = self.adaptive_mutation(self.mutation_factor, evaluations)
                mutant_vector = np.clip(x1 + adaptive_factor * (x2 - x3), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                    temperature = self.cooling_schedule(temperature, i)
                else:
                    prob_accept = np.exp((fitness[i] - trial_fitness) / temperature)
                    if np.random.rand() < prob_accept:
                        population[i] = trial_vector
                        fitness[i] = trial_fitness
                
                avg_fitness = np.mean(fitness)
                diversity = np.std(fitness)
                if diversity < self.diversity_control_factor * avg_fitness and evaluations < self.budget:
                    population[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    fitness[i] = func(population[i])
                    evaluations += 1

            temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]