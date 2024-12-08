import numpy as np

class SwarmGuidedStochasticSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.swarm_size = max(5, int(budget / (10 * dim)))  # Heuristic for swarm size
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        
    def __call__(self, func):
        # Initialize swarm
        swarm_positions = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        swarm_velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(swarm_positions)
        personal_best_fitness = np.array([func(ind) for ind in swarm_positions])
        num_evaluations = self.swarm_size

        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        while num_evaluations < self.budget:
            for i in range(self.swarm_size):
                if num_evaluations >= self.budget:
                    break
                
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                swarm_velocities[i] = (self.inertia_weight * swarm_velocities[i] +
                                       self.cognitive_weight * r1 * (personal_best_positions[i] - swarm_positions[i]) +
                                       self.social_weight * r2 * (global_best_position - swarm_positions[i]))
                
                # Update position
                swarm_positions[i] += swarm_velocities[i]
                swarm_positions[i] = np.clip(swarm_positions[i], self.lb, self.ub)
                
                # Stochastic perturbation
                perturbation = np.random.normal(0, 0.1, self.dim)
                candidate_position = np.clip(swarm_positions[i] + perturbation, self.lb, self.ub)
                
                # Evaluate candidate position
                candidate_fitness = func(candidate_position)
                num_evaluations += 1
                
                # Update personal best
                if candidate_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = candidate_position
                    personal_best_fitness[i] = candidate_fitness
                    
                    # Update global best
                    if candidate_fitness < global_best_fitness:
                        global_best_position = candidate_position
                        global_best_fitness = candidate_fitness
        
        return global_best_position, global_best_fitness