import numpy as np

class ImprovedDynamicPopSizeFireflyAlgorithm(EnhancedFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.pop_size = 10  # Initial population size
        self.alpha = 0.3  # Levy flight step size adaptation parameter
        
    def __call__(self, func):
        def improved_levy_flight():
            beta = self.beta0 / np.sqrt(self.dim)
            sigma1 = (np.prod(np.power(np.arange(1, self.dim+1), -beta)))**(1/self.dim)
            step = np.random.standard_normal(self.dim) * sigma1
            step_size = self.alpha * np.random.gamma(shape=2, scale=1)
            return step * step_size
        
        def attraction(x, y):
            r = np.linalg.norm(x - y)
            return np.exp(-self.gamma * r**2)
        
        def levy_update(x):
            step = improved_levy_flight()
            new_x = x + step * np.random.normal(0, 1, self.dim)
            return np.clip(new_x, self.lb, self.ub)
        
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(indiv) for indiv in pop])
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        step_size = attraction(pop[i], pop[j])
                        pop[i] = levy_update(pop[i]) if step_size > np.random.rand() else pop[i]
                        fitness[i] = func(pop[i])
            
            # Enhanced dynamic population size adaptation
            if np.random.rand() < 0.1:  # Probability of change
                self.pop_size = min(30, self.pop_size + 5)
                pop = np.vstack((pop, np.random.uniform(self.lb, self.ub, (5, self.dim)))
                fitness = np.append(fitness, [func(indiv) for indiv in pop[-5:]])
        
        return pop[np.argmin(fitness)]