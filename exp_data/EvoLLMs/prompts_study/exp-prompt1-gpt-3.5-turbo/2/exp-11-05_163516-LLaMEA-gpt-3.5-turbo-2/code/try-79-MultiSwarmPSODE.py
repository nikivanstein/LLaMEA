import numpy as np

class MultiSwarmPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 5
        self.swarm_size = 10
        self.max_iter = budget // (self.num_swarms * self.swarm_size)
        self.swarms = [np.random.uniform(-5.0, 5.0, (self.swarm_size, dim)) for _ in range(self.num_swarms)]
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.cr = 0.9
        self.f = 0.8
        
    def __call__(self, func):
        best_positions = [swarm[np.argmin([func(ind) for ind in swarm])] for swarm in self.swarms]
        best_fitnesses = [func(pos) for pos in best_positions]
        for _ in range(self.max_iter):
            for s in range(self.num_swarms):
                for i in range(self.swarm_size):
                    # PSO update
                    r1, r2 = np.random.uniform(0, 1, 2)
                    new_velocity = self.w * self.swarms[s][i] + self.c1 * r1 * (best_positions[s] - self.swarms[s][i]) + self.c2 * r2 * (best_positions[s] - self.swarms[s][i])
                    new_position = self.swarms[s][i] + new_velocity
                    new_position = np.clip(new_position, -5.0, 5.0)
                    
                    # DE update
                    rand_indexes = np.random.choice(np.arange(self.swarm_size), 3, replace=False)
                    mutant = self.swarms[s][rand_indexes[0]] + self.f * (self.swarms[s][rand_indexes[1]] - self.swarms[s][rand_indexes[2]])
                    crossover = np.random.rand(self.dim) < self.cr
                    trial = np.where(crossover, mutant, self.swarms[s][i])
                    
                    if func(trial) < func(self.swarms[s][i]):
                        self.swarms[s][i] = trial
                    if func(new_position) < func(self.swarms[s][i]):
                        self.swarms[s][i] = new_position
                    
                    # Update the best for the swarm
                    if func(self.swarms[s][i]) < best_fitnesses[s]:
                        best_positions[s] = self.swarms[s][i]
                        best_fitnesses[s] = func(best_positions[s])
                        
                        # Dynamic parameter adaptation
                        self.w = max(0.4, self.w * 0.99)
                        self.c1 = max(0.5, self.c1 * 0.99)
                        self.c2 = min(2.0, self.c2 * 1.01)
                        self.cr = min(1.0, self.cr * 1.01)
                        self.f = max(0.5, self.f * 0.99)
        
        return best_positions[np.argmin(best_fitnesses)]