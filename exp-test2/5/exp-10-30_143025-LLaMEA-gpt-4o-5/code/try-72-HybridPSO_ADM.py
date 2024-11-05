import numpy as np

class HybridPSO_ADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 20
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.mutation_coeff = 0.8
        self.vel_clamp = (-1.0, 1.0)
        
        self.positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.velocities = np.random.uniform(self.vel_clamp[0], self.vel_clamp[1], (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def __call__(self, func):
        global_best_position = None
        global_best_score = np.inf
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                score = func(self.positions[i])
                evaluations += 1
                if evaluations >= self.budget:
                    break
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                    
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = self.positions[i]

            self.inertia_weight = 0.5 + 0.2 * np.sin(2 * np.pi * evaluations / self.budget)
            self.social_coeff = 2.0 * (1 - evaluations / self.budget)  # Dynamic adjustment
            
            if evaluations % 10 == 0:  # Select a local leader every 10 evaluations
                local_leader_idx = np.random.choice(self.population_size)
                local_best_position = self.personal_best_positions[local_leader_idx]
            
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.social_coeff * r2 * (global_best_position - self.positions[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_component + social_component)
                self.vel_clamp = (-1.0 + 0.5 * (evaluations / self.budget), 1.0 - 0.5 * (evaluations / self.budget))  # Adaptive velocity clamp
                self.velocities[i] = np.clip(self.velocities[i], self.vel_clamp[0], self.vel_clamp[1])
                
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])

                neighbor_idx = np.random.choice(self.population_size)
                adaptive_weight = 0.1 + 0.1 * np.random.rand()
                neighbor_attraction = adaptive_weight * (self.personal_best_positions[neighbor_idx] - self.positions[i])
                self.positions[i] += neighbor_attraction * (1 - evaluations / self.budget)  # Dynamic neighborhood influence

                self.mutation_coeff = 0.4 + 0.4 * (1 - evaluations / self.budget)  # Dynamic mutation coefficient
                if np.random.rand() < self.mutation_coeff:
                    a, b, c = np.random.choice(self.population_size, 3, replace=False)
                    mutant_vector = self.positions[a] + 0.5 * (self.positions[b] - self.positions[c])
                    mutant_vector = np.clip(mutant_vector, self.bounds[0], self.bounds[1])
                    
                    trial_vector = np.where(np.random.rand(self.dim) < 0.5, mutant_vector, self.positions[i])
                    if func(trial_vector) < func(self.positions[i]):
                        self.positions[i] = trial_vector
            
        return global_best_position