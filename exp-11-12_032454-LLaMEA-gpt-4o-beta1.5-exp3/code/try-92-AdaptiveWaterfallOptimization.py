import numpy as np

class AdaptiveWaterfallOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.drops_count = max(5, int(budget / (10 * dim)))  # heuristic for number of drops
        self.initial_velocity = 0.5
        self.evaporate_factor = 0.95
        self.adapt_rate = 0.1
        
    def __call__(self, func):
        np.random.seed(42)  # Ensure reproducibility
        # Initialize water drops
        position = np.random.uniform(self.lb, self.ub, (self.drops_count, self.dim))
        velocity = np.random.uniform(-self.initial_velocity, self.initial_velocity, (self.drops_count, self.dim))
        personal_best_position = np.copy(position)
        personal_best_fitness = np.array([func(ind) for ind in position])
        num_evaluations = self.drops_count
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = position[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        while num_evaluations < self.budget:
            for i in range(self.drops_count):
                if num_evaluations >= self.budget:
                    break
                
                # Update velocity with adaptive adaptation
                velocity[i] *= self.evaporate_factor
                adapt = self.adapt_rate * np.random.rand(self.dim) * (global_best_position - position[i])
                velocity[i] += adapt
                
                # Move the drop
                new_position = position[i] + velocity[i]
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate the new position
                new_fitness = func(new_position)
                num_evaluations += 1
                
                # Update personal best
                if new_fitness < personal_best_fitness[i]:
                    personal_best_position[i] = new_position
                    personal_best_fitness[i] = new_fitness
                    
                    # Update global best
                    if new_fitness < global_best_fitness:
                        global_best_position = new_position
                        global_best_fitness = new_fitness
                
                # Update position
                position[i] = new_position
        
        return global_best_position, global_best_fitness