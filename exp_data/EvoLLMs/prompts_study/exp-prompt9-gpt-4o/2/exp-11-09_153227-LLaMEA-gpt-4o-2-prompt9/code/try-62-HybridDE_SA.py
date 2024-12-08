import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = max(4, 10 * dim)
        self.F = 0.8
        self.CR = 0.9
        self.temperature = 100

    def __call__(self, func):
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]
        
        evaluations = population_size

        while evaluations < self.budget:
            for i in range(population_size):
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                adaptive_F = self.F * (1 - evaluations / self.budget)
                mutant = population[a] + adaptive_F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                if np.random.rand() < 0.15:  # Reduced mutation probability
                    mutant = best + np.random.normal(0, 0.4, self.dim)  # Slightly narrower distribution

                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

            population_size = max(4, int(self.initial_population_size * (1 - evaluations / self.budget)))

            for i in range(population_size):
                candidate = population[i] + np.random.normal(0, 0.1, self.dim)
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                evaluations += 1

                acceptance_prob = np.exp(-(candidate_fitness - fitness[i]) / self.temperature)
                if candidate_fitness < fitness[i] or np.random.rand() < acceptance_prob:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < best_fitness:
                        best = candidate
                        best_fitness = candidate_fitness

            worst_idx = np.argmax(fitness)
            if fitness[worst_idx] > best_fitness:
                population[worst_idx] = best
                fitness[worst_idx] = best_fitness

            self.temperature *= 0.985  # Slightly faster cooling

        return best