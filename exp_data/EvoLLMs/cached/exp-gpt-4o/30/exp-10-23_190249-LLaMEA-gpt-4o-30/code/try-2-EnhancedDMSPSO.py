import numpy as np

class EnhancedDMSPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 40
        self.num_swarms = 5
        self.inertia = 0.9  # Initialize with higher inertia for exploration
        self.cognitive = 1.5
        self.social = 2.0
        self.global_best_position = None
        self.global_best_value = np.inf
        
    def __call__(self, func):
        np.random.seed(0)
        positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.num_particles, np.inf)
        
        for i in range(self.num_particles):
            value = func(positions[i])
            if value < personal_best_values[i]:
                personal_best_values[i] = value
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = np.copy(positions[i])
        
        eval_count = self.num_particles
        
        while eval_count < self.budget:
            self.inertia = 0.5 + np.random.rand() * 0.4  # Adapt inertia dynamically
            np.random.shuffle(positions)
            swarms = np.array_split(positions, self.num_swarms)
            
            for swarm in swarms:
                local_best_position = None
                local_best_value = np.inf
                
                for position in swarm:
                    value = func(position)
                    eval_count += 1
                    if value < local_best_value:
                        local_best_value = value
                        local_best_position = np.copy(position)
                    
                    idx = np.where((personal_best_positions == position).all(axis=1))[0][0]
                    if value < personal_best_values[idx]:
                        personal_best_values[idx] = value
                        personal_best_positions[idx] = np.copy(position)
                    
                    if eval_count >= self.budget:
                        break
                
                for idx in range(len(swarm)):
                    particle_idx = np.where((positions == swarm[idx]).all(axis=1))[0][0]
                    velocities[particle_idx] = (
                        self.inertia * velocities[particle_idx] +
                        self.cognitive * np.random.rand(self.dim) * 
                        (personal_best_positions[particle_idx] - swarm[idx]) +
                        self.social * np.random.rand(self.dim) * 
                        (local_best_position - swarm[idx])
                    )
                    perturbation = np.random.normal(0, 0.1, self.dim)  # Add local perturbation
                    positions[particle_idx] = positions[particle_idx] + velocities[particle_idx] + perturbation
                    positions[particle_idx] = np.clip(positions[particle_idx], self.lb, self.ub)
                
                if eval_count >= self.budget:
                    break
            
            for i in range(self.num_particles):
                value = func(positions[i])
                eval_count += 1
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = np.copy(positions[i])
                
                if eval_count >= self.budget:
                    break
        
        return self.global_best_position, self.global_best_value