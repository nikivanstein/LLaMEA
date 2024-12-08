import numpy as np

class AdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))
        self.w = 0.9  # inertia weight
        self.w_min = 0.4
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.v_max = (self.ub - self.lb) * 0.2  # max velocity
        self.v_min = -self.v_max
        
    def __call__(self, func):
        # Initialize swarm
        positions = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(self.v_min, self.v_max, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.array([func(pos) for pos in positions])
        num_evaluations = self.population_size
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] + 
                                self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                self.c2 * r2 * (global_best_position - positions[i]))
                velocities[i] = np.clip(velocities[i], self.v_min, self.v_max)
                
                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)
                
                # Evaluate fitness
                fitness = func(positions[i])
                num_evaluations += 1
                
                # Update personal and global best
                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best_position = positions[i]
                        global_best_fitness = fitness
            
            # Dynamic inertia weight adjustment
            self.w = self.w_min + (0.5 * (self.budget - num_evaluations) / self.budget)
        
        return global_best_position, global_best_fitness