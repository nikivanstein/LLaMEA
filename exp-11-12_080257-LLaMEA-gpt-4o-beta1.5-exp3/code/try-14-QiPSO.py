import numpy as np

class QiPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 3 * dim)
        self.eval_count = 0
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.position = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = self.position.copy()
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_fitness = np.inf

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.position])
        self.eval_count += len(self.position)
        return fitness

    def update_personal_best(self, fitness):
        better_mask = fitness < self.personal_best_fitness
        self.personal_best[better_mask] = self.position[better_mask]
        self.personal_best_fitness[better_mask] = fitness[better_mask]

    def update_global_best(self):
        min_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[min_idx] < self.global_best_fitness:
            self.global_best_fitness = self.personal_best_fitness[min_idx]
            self.global_best = self.personal_best[min_idx].copy()

    def quantum_behavior(self):
        for i in range(self.population_size):
            if np.random.rand() < 0.5:
                self.position[i] = self.global_best + np.random.uniform(-1, 1, self.dim) * np.abs(self.position[i] - self.global_best)
            else:
                self.position[i] = self.personal_best[i] + np.random.uniform(-1, 1, self.dim) * np.abs(self.position[i] - self.personal_best[i])
            self.position[i] = np.clip(self.position[i], self.lower_bound, self.upper_bound)

    def update_velocity_and_position(self):
        for i in range(self.population_size):
            inertia = self.inertia_weight * self.velocity[i]
            cognitive = self.cognitive_coeff * np.random.rand(self.dim) * (self.personal_best[i] - self.position[i])
            social = self.social_coeff * np.random.rand(self.dim) * (self.global_best - self.position[i])
            self.velocity[i] = inertia + cognitive + social
            self.position[i] = self.position[i] + self.velocity[i]
            self.position[i] = np.clip(self.position[i], self.lower_bound, self.upper_bound)

    def adaptive_local_search(self, func):
        epsilon = 0.1 * (self.upper_bound - self.lower_bound)
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break
            neighbor = np.clip(self.position[i] + epsilon * np.random.uniform(-1, 1, self.dim), self.lower_bound, self.upper_bound)
            neighbor_fitness = func(neighbor)
            self.eval_count += 1
            if neighbor_fitness < self.personal_best_fitness[i]:
                self.personal_best[i] = neighbor
                self.personal_best_fitness[i] = neighbor_fitness

    def __call__(self, func):
        fitness = self.evaluate_population(func)
        self.update_personal_best(fitness)
        self.update_global_best()

        while self.eval_count < self.budget:
            self.update_velocity_and_position()
            fitness = self.evaluate_population(func)
            self.update_personal_best(fitness)
            self.update_global_best()
            self.quantum_behavior()
            self.adaptive_local_search(func)

        return self.global_best