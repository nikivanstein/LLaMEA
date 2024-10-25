import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 15)  # Adjusted for balance between exploration and exploitation
        self.inertia_weight = 0.35 + np.random.rand() * 0.45  # Fine-tuned for dynamic adjustment
        self.cognitive_coeff = 0.8 + np.random.rand() * 0.9  # Broadened for better individual focus
        self.social_coeff = 1.0 + np.random.rand() * 0.9  # Maintained for reliable convergence
        self.F = 0.5 + np.random.rand() * 0.3  # Further broadening of scaling factor for better mutation
        self.CR = 0.85  # Slightly reduced crossover probability for controlled diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.best_position = None
        self.best_value = float('inf')

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-0.15, 0.15, (self.population_size, self.dim))  # Slightly increased velocity range
        personal_best_positions = np.copy(positions)
        personal_best_values = np.array([func(pos) for pos in personal_best_positions])
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Differential Evolution Mutation and Crossover
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = positions[a] + self.F * (positions[b] - positions[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                trial_vector = np.copy(positions[i])
                
                if np.random.rand() < 0.15:  # Probabilistic decision to enhance diversity
                    trial_vector = mutant_vector
                else:
                    for j in range(self.dim):
                        if np.random.rand() < self.CR:
                            trial_vector[j] = mutant_vector[j]
                
                trial_value = func(trial_vector)
                evaluations += 1
                
                if trial_value < personal_best_values[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_values[i] = trial_value
                    
                    if trial_value < global_best_value:
                        global_best_position = trial_vector
                        global_best_value = trial_value
            
            # Particle Swarm Optimization update with enhanced interaction
            elite_indices = np.argsort(personal_best_values)[:max(3, self.population_size // 8)]  # Increased elite selection
            elite_positions = personal_best_positions[elite_indices]
            for i in range(self.population_size):
                elite_partner = elite_positions[np.random.choice(len(elite_positions))]
                r1, r2, r3 = np.random.rand(self.dim), np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                                self.social_coeff * r2 * (global_best_position - positions[i]) +
                                0.6 * r3 * (elite_partner - positions[i]))  # Enhanced influence from elite partners
                positions[i] += velocities[i]
                
                positions = np.clip(positions, self.lower_bound, self.upper_bound)
                
                current_value = func(positions[i])
                evaluations += 1
                
                if current_value < personal_best_values[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_values[i] = current_value
                    
                    if current_value < global_best_value:
                        global_best_position = positions[i]
                        global_best_value = current_value

        self.best_position = global_best_position
        self.best_value = global_best_value
        return self.best_position, self.best_value