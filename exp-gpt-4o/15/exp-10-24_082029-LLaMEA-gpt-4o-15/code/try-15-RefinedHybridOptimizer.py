import numpy as np

class RefinedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(80, budget // 10)  # Increased population size for diversity
        self.inertia_weight = 0.5 + np.random.rand() * 0.4  # Narrower adaptive inertia weight
        self.cognitive_coeff = 1.2 + np.random.rand() * 0.6  # Enhanced cognitive coefficient
        self.social_coeff = 1.2 + np.random.rand() * 0.6  # Enhanced social coefficient
        self.F = 0.7  # Adjusted scaling factor for better exploitation
        self.CR = 0.9  # Increased crossover probability for better convergence
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.best_position = None
        self.best_value = float('inf')
        self.restart_threshold = budget // 5  # Dynamic restarting mechanism
    
    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.array([func(pos) for pos in personal_best_positions])
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]
        
        evaluations = self.population_size
        last_improvement = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Differential Evolution Mutation and Crossover
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = positions[a] + self.F * (positions[b] - positions[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                trial_vector = np.copy(positions[i])
                
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
                        last_improvement = evaluations
            
            # Particle Swarm Optimization update with boundary adaptation
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                                self.social_coeff * r2 * (global_best_position - positions[i]))
                positions[i] += velocities[i]
                
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)
                
                current_value = func(positions[i])
                evaluations += 1
                
                if current_value < personal_best_values[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_values[i] = current_value
                    
                    if current_value < global_best_value:
                        global_best_position = positions[i]
                        global_best_value = current_value
                        last_improvement = evaluations
            
            # Dynamic restarting mechanism
            if evaluations - last_improvement > self.restart_threshold:
                positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                velocities = np.random.uniform(-0.3, 0.3, (self.population_size, self.dim))
                last_improvement = evaluations
        
        self.best_position = global_best_position
        self.best_value = global_best_value
        return self.best_position, self.best_value