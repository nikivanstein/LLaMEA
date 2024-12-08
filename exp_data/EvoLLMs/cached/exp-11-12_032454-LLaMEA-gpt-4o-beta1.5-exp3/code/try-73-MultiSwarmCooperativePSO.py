import numpy as np

class MultiSwarmCooperativePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.swarm_size = max(5, int(budget / (10 * dim)))
        self.num_swarms = max(2, dim // 5)  # Adapt number of swarms based on dimensionality
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w = 0.7   # inertia weight
        
    def __call__(self, func):
        # Initialize swarms
        swarms = [np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        fitness = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        num_evaluations = self.swarm_size * self.num_swarms
        
        personal_best_pos = [swarm.copy() for swarm in swarms]
        personal_best_fitness = [fit.copy() for fit in fitness]
        
        global_best_pos = np.array([swarm[np.argmin(fit)] for swarm, fit in zip(swarms, fitness)])
        global_best_fitness = np.array([np.min(fit) for fit in fitness])
        overall_best_idx = np.argmin(global_best_fitness)
        overall_best_pos = global_best_pos[overall_best_idx]
        overall_best_fitness = global_best_fitness[overall_best_idx]
        
        while num_evaluations < self.budget:
            for s_idx in range(self.num_swarms):
                for i in range(self.swarm_size):
                    if num_evaluations >= self.budget:
                        break
                    
                    # Update velocity
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    velocities[s_idx][i] = (
                        self.w * velocities[s_idx][i] +
                        self.c1 * r1 * (personal_best_pos[s_idx][i] - swarms[s_idx][i]) +
                        self.c2 * r2 * (global_best_pos[s_idx] - swarms[s_idx][i])
                    )
                    
                    # Update position
                    swarms[s_idx][i] += velocities[s_idx][i]
                    swarms[s_idx][i] = np.clip(swarms[s_idx][i], self.lb, self.ub)

                    # Evaluate fitness
                    current_fitness = func(swarms[s_idx][i])
                    num_evaluations += 1
                    
                    # Update personal and global bests
                    if current_fitness < personal_best_fitness[s_idx][i]:
                        personal_best_pos[s_idx][i] = swarms[s_idx][i]
                        personal_best_fitness[s_idx][i] = current_fitness
                    
                    if current_fitness < global_best_fitness[s_idx]:
                        global_best_pos[s_idx] = swarms[s_idx][i]
                        global_best_fitness[s_idx] = current_fitness
                        
                        if current_fitness < overall_best_fitness:
                            overall_best_pos = swarms[s_idx][i]
                            overall_best_fitness = current_fitness
        
        return overall_best_pos, overall_best_fitness