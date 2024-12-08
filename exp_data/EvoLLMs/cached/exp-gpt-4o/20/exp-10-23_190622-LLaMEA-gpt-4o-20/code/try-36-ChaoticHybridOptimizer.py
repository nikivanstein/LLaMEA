import numpy as np

class ChaoticHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 5 * dim
        self.mutation_factor = 0.70  # Slight tweak for exploration-exploitation trade-off
        self.crossover_rate = 0.80  # Reduced to control diversity better
        self.local_search_intensity = 5  # Rebalanced for local enhancement
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0
        self.chaotic_sequence = self.chaotic_map_sequence(self.budget, 0.7)

    def chaotic_map_sequence(self, length, init_value):
        x = init_value
        sequence = []
        for _ in range(length):
            x = 4.0 * x * (1 - x)  # Logistic map for chaos
            sequence.append(x)
        return sequence

    def differential_evolution(self, func):
        for i in range(self.population_size):
            if self.used_budget >= self.budget:
                break
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            chaos_idx = self.used_budget % len(self.chaotic_sequence)
            chaos_factor = self.chaotic_sequence[chaos_idx]
            mutant_vector = np.clip(a + chaos_factor * self.mutation_factor * (b - c), -5, 5)
            crossover = np.random.rand(self.dim) < self.crossover_rate
            trial_vector = np.where(crossover, mutant_vector, self.population[i])
            trial_fitness = func(trial_vector)
            self.used_budget += 1
            if trial_fitness < self.fitness[i]:
                self.fitness[i] = trial_fitness
                self.population[i] = trial_vector

    def stochastic_local_search(self, func):
        best_indices = np.argsort(self.fitness)[:self.local_search_intensity]
        for idx in best_indices:
            if self.used_budget >= self.budget:
                break
            step_size = np.random.uniform(0.05, 0.15)  # Narrower perturbation for local refinement
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(self.population[idx] + perturbation, -5, 5)
            candidate_fitness = func(candidate)
            self.used_budget += 1
            if candidate_fitness < self.fitness[idx]:
                self.fitness[idx] = candidate_fitness
                self.population[idx] = candidate

    def adapt_parameters(self):
        if np.std(self.fitness) < 0.3:  # Adjusted threshold for parameter adaptation
            self.local_search_intensity = 7  # Further increase local search focus
            self.mutation_factor = 0.80  # Tweak mutation for exploration
            self.crossover_rate = 0.85  # Further manage exploration-exploitation

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.used_budget = self.population_size
        while self.used_budget < self.budget:
            self.differential_evolution(func)
            self.stochastic_local_search(func)
            self.adapt_parameters()
        best_index = np.argmin(self.fitness)
        return self.population[best_index]