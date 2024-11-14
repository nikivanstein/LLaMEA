import numpy as np

class AdaptiveMultiSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_swarms = max(3, int(dim / 5))  # number of sub-swarms
        self.swarm_size = max(5, int(budget / (20 * dim)))  # population size per swarm
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.global_best = None
        self.global_best_fitness = float('inf')
        
    def __call__(self, func):
        np.random.seed(42)
        swarms = [self.initialize_swarm() for _ in range(self.num_swarms)]
        fitnesses = [np.array([func(p) for p in swarm['positions']]) for swarm in swarms]
        num_evaluations = self.num_swarms * self.swarm_size
        
        self.update_global_best(swarms, fitnesses)
        
        while num_evaluations < self.budget:
            for swarm_idx, swarm in enumerate(swarms):
                if num_evaluations >= self.budget:
                    break
                
                r1 = np.random.rand(self.swarm_size, self.dim)
                r2 = np.random.rand(self.swarm_size, self.dim)
                
                swarm['velocities'] = self.w * swarm['velocities'] + \
                    self.c1 * r1 * (swarm['best_positions'] - swarm['positions']) + \
                    self.c2 * r2 * (self.global_best - swarm['positions'])
                
                swarm['positions'] += swarm['velocities']
                swarm['positions'] = np.clip(swarm['positions'], self.lb, self.ub)
                
                new_fitnesses = np.array([func(p) for p in swarm['positions']])
                num_evaluations += self.swarm_size
                
                improved = new_fitnesses < fitnesses[swarm_idx]
                swarm['best_positions'][improved] = swarm['positions'][improved]
                fitnesses[swarm_idx][improved] = new_fitnesses[improved]
                
                self.update_swarm_best(swarm, fitnesses[swarm_idx])
                self.update_global_best(swarms, fitnesses)

            self.adapt_parameters()  # Adapt parameters dynamically based on progress
        
        return self.global_best, self.global_best_fitness
    
    def initialize_swarm(self):
        positions = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        best_positions = positions.copy()
        return {'positions': positions, 'velocities': velocities, 'best_positions': best_positions}
    
    def update_swarm_best(self, swarm, fitness):
        best_idx = np.argmin(fitness)
        swarm['local_best'] = swarm['best_positions'][best_idx]
        swarm['local_best_fitness'] = fitness[best_idx]
        
    def update_global_best(self, swarms, fitnesses):
        for swarm, fitness in zip(swarms, fitnesses):
            if swarm['local_best_fitness'] < self.global_best_fitness:
                self.global_best = swarm['local_best']
                self.global_best_fitness = swarm['local_best_fitness']
    
    def adapt_parameters(self):
        # Simple adaptive strategy, can be enhanced further
        self.w = 0.4 + 0.5 * np.random.rand()  # Inertia weight adaptation
        self.c1 = 1.5 + np.random.rand()  # Cognitive coefficient adaptation
        self.c2 = 1.5 + np.random.rand()  # Social coefficient adaptation