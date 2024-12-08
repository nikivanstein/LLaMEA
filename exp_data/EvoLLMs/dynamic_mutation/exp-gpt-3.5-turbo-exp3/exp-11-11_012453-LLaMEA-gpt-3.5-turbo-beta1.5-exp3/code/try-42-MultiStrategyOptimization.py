import numpy as np
from DifferentialEvolution import DifferentialEvolution

class MultiStrategyOptimization(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.de_iterations = 3  # Number of DE iterations to perform
        
    def __call__(self, func):
        de_optimizer = DifferentialEvolution(self.budget, self.dim)
        
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(indiv) for indiv in pop])
        
        for _ in range(self.budget // self.de_iterations):
            pop = de_optimizer(func, pop, fitness)
            
            for _ in range(self.de_iterations):
                for i in range(self.pop_size):
                    for j in range(self.pop_size):
                        if fitness[j] < fitness[i]:
                            step_size = attraction(pop[i], pop[j])
                            pop[i] = levy_update(pop[i]) if step_size > np.random.rand() else pop[i]
                            fitness[i] = func(pop[i])
                
                # Dynamic population size adaptation
                if np.random.rand() < 0.1:  # Probability of change
                    self.pop_size = min(30, self.pop_size + 5)
                    pop = np.vstack((pop, np.random.uniform(self.lb, self.ub, (5, self.dim)))
                    fitness = np.append(fitness, [func(indiv) for indiv in pop[-5:]])
        
        return pop[np.argmin(fitness)]