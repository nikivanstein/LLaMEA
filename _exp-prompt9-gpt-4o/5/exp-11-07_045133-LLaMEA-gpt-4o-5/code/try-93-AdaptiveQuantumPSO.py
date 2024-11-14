import numpy as np

class AdaptiveQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.final_population_size = 22
        self.population_size = self.initial_population_size
        self.phi = 0.46  # Slight adjustment to phi for enhanced quantum effect
        self.cognitive_weight = 1.47  # Slightly increased cognitive weight for individual emphasis
        self.social_weight = 1.53  # Reduced social weight to maintain balance
        self.inertia_weight = 0.82  # Further reduced inertia for quicker adaptation
        self.position = np.random.uniform(-5, 5, (self.population_size, dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                fitness = func(self.position[i])
                evaluations += 1

                if fitness < self.personal_best_value[i]:
                    self.personal_best_value[i] = fitness
                    self.personal_best_position[i] = self.position[i]

                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.position[i]

            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                cognitive_component = self.cognitive_weight * r1 * (self.personal_best_position[i] - self.position[i])
                social_component = self.social_weight * r2 * (self.global_best_position - self.position[i])
                quantum_effect = np.sign(np.random.uniform(-1, 1, self.dim)) * self.phi * np.log(np.clip(1/np.random.rand(), 1e-5, 1e5))
                self.velocity[i] = (self.inertia_weight * self.velocity[i] + cognitive_component + social_component + quantum_effect)
                self.position[i] = np.clip(self.position[i] + self.velocity[i], -5, 5)

            self.population_size = max(self.final_population_size, self.initial_population_size - evaluations // (self.budget // 14))
            self.position = self.position[:self.population_size]
            self.velocity = self.velocity[:self.population_size]
            self.personal_best_position = self.personal_best_position[:self.population_size]
            self.personal_best_value = self.personal_best_value[:self.population_size]