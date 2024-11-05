import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.temp = 1000.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.99  # Cooling rate for simulated annealing

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        num_evaluations = self.pop_size

        def mutate(target_idx):
            indices = [idx for idx in range(self.pop_size) if idx != target_idx]
            a, b, c = np.random.choice(indices, 3, replace=False)
            dynamic_F = self.F * (1.0 - (num_evaluations / self.budget)) * np.exp(-num_evaluations / self.budget)  # Decay factor added
            mutant = population[a] + dynamic_F * (population[b] - population[c])
            mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
            return mutant

        while num_evaluations < self.budget:
            self.CR = 0.9 - (0.5 * (num_evaluations / self.budget))  # Dynamic adjustment with enhanced exploration
            for i in range(self.pop_size):
                if num_evaluations >= self.budget:
                    break
                mutant = mutate(i)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial

            # Simulated Annealing part
            for i in range(self.pop_size):
                if num_evaluations >= self.budget:
                    break
                neighbor = population[i] + np.random.normal(0, 0.1 * self.temp / 1000, self.dim)
                neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
                neighbor_fitness = func(neighbor)
                num_evaluations += 1
                delta_fit = neighbor_fitness - fitness[i]

                if delta_fit < 0 or np.random.rand() < np.exp(-delta_fit / self.temp):
                    population[i] = neighbor
                    fitness[i] = neighbor_fitness
                    if neighbor_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = neighbor

            self.temp *= (self.cooling_rate - 0.01 * (num_evaluations / self.budget))  # Adjusted cooling rate
            self.pop_size = max(4, int(8 * self.dim * (1 - num_evaluations / self.budget)))  # Adaptive population size

        return best_solution