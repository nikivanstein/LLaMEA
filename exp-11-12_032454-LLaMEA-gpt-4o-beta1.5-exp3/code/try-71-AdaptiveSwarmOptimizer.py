import numpy as np

class AdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(20, int(budget / (5 * dim)))  # heuristic for population size
        self.inertia_weight = 0.7  # initial inertia weight
        self.inertia_damp = 0.99  # inertia weight damping factor
        self.cognitive_coeff = 1.5  # personal attraction factor
        self.social_coeff = 1.5  # global attraction factor
    
    def __call__(self, func):
        # Initialize particle positions and velocities
        positions = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.array([func(ind) for ind in positions])
        num_evaluations = self.population_size
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break

                # Update velocities
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.social_coeff * r2 * (global_best_position - positions[i])
                velocities[i] = (self.inertia_weight * velocities[i] + cognitive_velocity + social_velocity)
                velocities[i] = np.clip(velocities[i], self.lb - positions[i], self.ub - positions[i])
                
                # Update positions
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)
                
                # Evaluate new position
                current_fitness = func(positions[i])
                num_evaluations += 1
                
                # Update personal best
                if current_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = current_fitness
                    
                    # Update global best
                    if current_fitness < global_best_fitness:
                        global_best_position = positions[i]
                        global_best_fitness = current_fitness
            
            # Dampen inertia weight
            self.inertia_weight *= self.inertia_damp

        return global_best_position, global_best_fitness