import numpy as np

class SwarmBasedQuantumParticleOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.swarm_size = max(5, int(budget / (10 * dim)))  # heuristic for swarm size
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        
    def __call__(self, func):
        # Initialize particles
        position = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        quantum_position = np.random.uniform(0, 1, (self.swarm_size, self.dim))
        
        personal_best_position = np.copy(position)
        personal_best_fitness = np.array([func(ind) for ind in position])
        num_evaluations = self.swarm_size
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = position[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        while num_evaluations < self.budget:
            for i in range(self.swarm_size):
                if num_evaluations >= self.budget:
                    break
                
                # Update velocity
                inertia = self.inertia_weight * velocity[i]
                cognitive = self.cognitive_constant * np.random.rand(self.dim) * (personal_best_position[i] - position[i])
                social = self.social_constant * np.random.rand(self.dim) * (global_best_position - position[i])
                
                new_velocity = inertia + cognitive + social
                new_position = position[i] + new_velocity
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Quantum effect
                quantum_effect = np.random.rand(self.dim) < quantum_position[i]
                new_position = np.where(quantum_effect, new_position, global_best_position)
                
                # Evaluate new position
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
                
                # Update particle position and velocity
                position[i] = new_position
                velocity[i] = new_velocity
                quantum_position[i] = np.random.uniform(0, 1, self.dim)  # Update quantum state
        
        return global_best_position, global_best_fitness