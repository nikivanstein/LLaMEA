import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim  # Common heuristic for DE
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 1000.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.99

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound,
                                 (self.population_size, self.dim))

    def _mutate(self, idx, population):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = np.clip(population[a] + self.mutation_factor * (population[b] - population[c]),
                         self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        crossover_mask[np.random.randint(0, self.dim)] = True  # Ensure at least one element is from mutant
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _acceptance_probability(self, old_cost, new_cost, temperature):
        if new_cost < old_cost:
            return 1.0
        else:
            return np.exp((old_cost - new_cost) / temperature)

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self._mutate(i, population)
                trial = self._crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1
                if evaluations >= self.budget:
                    break

                # Simulated Annealing acceptance criteria
                if trial_fitness < fitness[i] or np.random.rand() < self._acceptance_probability(fitness[i], trial_fitness, self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_solution = trial.copy()
                    best_fitness = trial_fitness

            self.temperature *= self.cooling_rate  # Cool down the temperature

        return best_solution