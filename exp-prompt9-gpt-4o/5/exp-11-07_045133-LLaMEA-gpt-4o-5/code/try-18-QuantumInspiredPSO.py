import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.final_population_size = 18  # Slightly reduced for quicker focus
        self.population_size = self.initial_population_size
        self.phi = 0.55  # Adjusted for stronger quantum influence
        self.cognitive_weight = 1.7  # Slight increase for better self-exploration
        self.social_weight = 1.4  # Slight decrease to balance social influence
        self.inertia_weight = 0.88  # Fine-tuned for improved stability
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
                quantum_effect = np.sign(np.random.uniform(-1, 1, self.dim)) * self.phi * np.log(1/np.random.rand())
                self.velocity[i] = (self.inertia_weight * self.velocity[i] + cognitive_component + social_component + quantum_effect)
                self.position[i] = np.clip(self.position[i] + self.velocity[i], -5, 5)
            
            # Dynamic population adjustment
            self.population_size = max(self.final_population_size, self.initial_population_size - evaluations // (self.budget // 11))  # Slightly slower reduction
            self.position = self.position[:self.population_size]
            self.velocity = self.velocity[:self.population_size]
            self.personal_best_position = self.personal_best_position[:self.population_size]
            self.personal_best_value = self.personal_best_value[:self.population_size]