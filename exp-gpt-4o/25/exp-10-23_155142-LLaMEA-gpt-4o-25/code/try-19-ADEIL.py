import numpy as np

class ADEIL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = max(10, 10 * dim)  # Ensure a minimum population size
        self.F = 0.5  # Differential weight
        self.Cr = 0.9  # Crossover probability
        self.memory = []
        self.inertia = 0.7  # Inertia factor for mutation

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size
        historical_best = np.copy(population[np.argmin(fitness)])

        while eval_count < self.budget:
            if len(self.memory) > 0:
                self.F = np.mean([entry[0] for entry in self.memory])
                self.Cr = np.mean([entry[1] for entry in self.memory])

            new_population = np.copy(population)
            for i in range(self.pop_size):
                indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.inertia * self.F * (x2 - x3), self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < self.Cr
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.memory.append((self.F, self.Cr))
                    if len(self.memory) > 50:
                        self.memory.pop(0)

                if eval_count >= self.budget:
                    break

            population = new_population
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < func(historical_best):
                historical_best = np.copy(population[best_idx])

        return historical_best, func(historical_best)