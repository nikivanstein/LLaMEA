import numpy as np

class HybridAdaptivePSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4 * dim, 20)
        self.inertia_weight = 0.5
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.mutation_factor = 0.6
        self.crossover_rate = 0.85
        self.local_search_intensity = 6
        self.adaptive_threshold = 0.15
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_fitness = np.inf
        self.used_budget = 0

    def particle_swarm_optimization(self, func):
        for i in range(self.population_size):
            if self.used_budget >= self.budget:
                break
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            self.velocity[i] = (self.inertia_weight * self.velocity[i] +
                                self.cognitive_component * r1 * (self.personal_best_positions[i] - self.population[i]) +
                                self.social_component * r2 * (self.global_best_position - self.population[i]))
            self.population[i] = np.clip(self.population[i] + self.velocity[i], -5, 5)
            fitness = func(self.population[i])
            self.used_budget += 1
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best_positions[i] = self.population[i]
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.population[i]

    def differential_mutation(self, func):
        for i in range(self.population_size):
            if self.used_budget >= self.budget:
                break
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant_vector = np.clip(a + self.mutation_factor * (b - c), -5, 5)
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
            step_size = np.random.uniform(0.03, 0.15)
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(self.population[idx] + perturbation, -5, 5)
            candidate_fitness = func(candidate)
            self.used_budget += 1
            if candidate_fitness < self.fitness[idx]:
                self.fitness[idx] = candidate_fitness
                self.population[idx] = candidate

    def adapt_parameters(self):
        if np.std(self.fitness) < self.adaptive_threshold:
            self.local_search_intensity = 8
            self.mutation_factor = 0.7
            self.crossover_rate = 0.88

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.personal_best_fitness = np.copy(self.fitness)
        self.personal_best_positions = np.copy(self.population)
        self.global_best_position = self.population[np.argmin(self.fitness)]
        self.global_best_fitness = np.min(self.fitness)
        self.used_budget = self.population_size
        while self.used_budget < self.budget:
            self.particle_swarm_optimization(func)
            self.differential_mutation(func)
            self.stochastic_local_search(func)
            self.adapt_parameters()
        return self.global_best_position