import numpy as np

class HAMsPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 40
        self.num_swarms = 5
        self.inertia = 0.7
        self.cognitive = 1.5
        self.social = 2.0
        self.global_best_position = None
        self.global_best_value = np.inf
        
    def __call__(self, func):
        np.random.seed(0)
        # Initialize particles' positions and velocities
        positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.num_particles, np.inf)
        
        # Evaluate initial positions
        for i in range(self.num_particles):
            value = func(positions[i])
            personal_best_values[i] = value
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = np.copy(positions[i])
        
        eval_count = self.num_particles
        
        while eval_count < self.budget:
            # Dynamically adjust swarms based on performance
            adaptive_num_swarms = np.random.randint(3, self.num_swarms + 1)
            np.random.shuffle(positions)
            swarms = np.array_split(positions, adaptive_num_swarms)
            
            for swarm in swarms:
                local_best_position = None
                local_best_value = np.inf
                
                for position in swarm:
                    value = func(position)
                    eval_count += 1
                    if value < local_best_value:
                        local_best_value = value
                        local_best_position = np.copy(position)
                    
                    idx = np.where((personal_best_positions == position).all(axis=1))[0]
                    if idx.size > 0 and value < personal_best_values[idx[0]]:
                        personal_best_values[idx[0]] = value
                        personal_best_positions[idx[0]] = np.copy(position)
                    
                    if eval_count >= self.budget:
                        break
                
                # Update velocities and positions with elitist approach
                for idx in range(len(swarm)):
                    particle_idx = np.where((positions == swarm[idx]).all(axis=1))[0]
                    if particle_idx.size == 0:
                        continue
                    particle_idx = particle_idx[0]
                    velocities[particle_idx] = (
                        self.inertia * velocities[particle_idx] +
                        self.cognitive * np.random.rand(self.dim) * 
                        (personal_best_positions[particle_idx] - swarm[idx]) +
                        self.social * np.random.rand(self.dim) * 
                        (local_best_position - swarm[idx])
                    )
                    positions[particle_idx] = positions[particle_idx] + velocities[particle_idx]
                    positions[particle_idx] = np.clip(positions[particle_idx], self.lb, self.ub)
                
                if eval_count >= self.budget:
                    break
            
            # Merge the best performing swarms and update global best
            best_swarm = min(swarms, key=lambda swarm: min(func(pos) for pos in swarm))
            for position in best_swarm:
                value = func(position)
                eval_count += 1
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = np.copy(position)
                
                if eval_count >= self.budget:
                    break
        
        return self.global_best_position, self.global_best_value