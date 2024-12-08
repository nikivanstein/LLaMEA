import numpy as np

class MultiSwarmDynamicPopSizeFireflyAlgorithm(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.num_swarms = 5
        self.swarm_size = 10
        
    def __call__(self, func):
        def levy_flight():
            beta = self.beta0 / np.sqrt(self.dim)
            sigma1 = (np.prod(np.power(np.arange(1, self.dim+1), -beta)))**(1/self.dim)
            sigma2 = np.power(np.random.standard_normal(self.dim) * sigma1, 1/beta)
            return sigma2
        
        def attraction(x, y):
            r = np.linalg.norm(x - y)
            return np.exp(-self.gamma * r**2)
        
        def levy_update(x):
            step = levy_flight()
            new_x = x + step * np.random.normal(0, 1, self.dim)
            return np.clip(new_x, self.lb, self.ub)
        
        # Initialize multiple swarms
        swarms = [np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        swarm_fitness = [np.array([func(indiv) for indiv in swarm]) for swarm in swarms]
        
        for _ in range(self.budget):
            for swarm_idx in range(self.num_swarms):
                swarm = swarms[swarm_idx]
                fitness = swarm_fitness[swarm_idx]
                
                for i in range(self.swarm_size):
                    for j in range(self.swarm_size):
                        if fitness[j] < fitness[i]:
                            step_size = attraction(swarm[i], swarm[j])
                            swarm[i] = levy_update(swarm[i]) if step_size > np.random.rand() else swarm[i]
                            fitness[i] = func(swarm[i])
                
                # Dynamic population size adaptation per swarm
                if np.random.rand() < 0.1:
                    self.swarm_size = min(30, self.swarm_size + 5)
                    swarm = np.vstack((swarm, np.random.uniform(self.lb, self.ub, (5, self.dim))))
                    fitness = np.append(fitness, [func(indiv) for indiv in swarm[-5:]])
                
                swarms[swarm_idx] = swarm
                swarm_fitness[swarm_idx] = fitness
        
        # Select the best individual across all swarms
        all_pop = np.concatenate(swarms)
        all_fitness = np.concatenate(swarm_fitness)
        best_idx = np.argmin(all_fitness)
        
        return all_pop[best_idx]