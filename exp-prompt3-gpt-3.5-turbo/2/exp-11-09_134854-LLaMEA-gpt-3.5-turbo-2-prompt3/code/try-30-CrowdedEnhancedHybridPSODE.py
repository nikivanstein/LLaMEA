import numpy as np

class CrowdedEnhancedHybridPSODE(EnhancedHybridPSODE):
    def __init__(self, budget, dim, pop_size=30, w=0.5, c1=1.5, c2=1.5, f=0.5, cr=0.9, adapt_rate=0.1, crowding_factor=0.1):
        super().__init__(budget, dim, pop_size, w, c1, c2, f, cr, adapt_rate)
        self.crowding_factor = crowding_factor

    def __call__(self, func):
        def crowding_selection(population, fitness):
            crowding_values = np.zeros(len(population))
            for d in range(self.dim):
                sorted_indices = np.argsort(population[:, d])
                crowding_values[sorted_indices[0]] += self.crowding_factor
                crowding_values[sorted_indices[-1]] += self.crowding_factor
                for i in range(1, len(population) - 1):
                    crowding_values[sorted_indices[i]] += (population[sorted_indices[i + 1], d] - population[sorted_indices[i - 1], d]) / (population[sorted_indices[-1], d] - population[sorted_indices[0], d])
            return crowding_values

        population = initialize_population()
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        for _ in range(self.budget - self.pop_size):
            crowding_values = crowding_selection(population, fitness)
            sorted_indices = np.argsort(crowding_values)[::-1]
            for i in range(self.pop_size):
                r1, r2, r3 = np.random.choice(sorted_indices[:self.pop_size], 3, replace=False)
                mutant = population[r1] + self.f * (population[r2] - population[r3])
                self.f = max(0.1, min(0.9, self.f + np.random.normal(0, self.adapt_rate)))
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial

            for i in range(self.pop_size):
                r1, r2 = np.random.choice(sorted_indices[:self.pop_size], 2, replace=False)
                v = self.w * population[i] + self.c1 * np.random.rand(self.dim) * (best_solution - population[i]) + self.c2 * np.random.rand(self.dim) * (population[r1] - population[r2])
                mutation_direction = np.random.choice([-1, 1], p=[self.mut_prob, 1 - self.mut_prob])
                self.mut_prob = max(0.1, min(0.9, self.mut_prob + np.random.normal(0, self.adapt_rate)))  # Dynamically adjust mutation probability
                population[i] = np.clip(v, -5.0, 5.0)

        return best_solution