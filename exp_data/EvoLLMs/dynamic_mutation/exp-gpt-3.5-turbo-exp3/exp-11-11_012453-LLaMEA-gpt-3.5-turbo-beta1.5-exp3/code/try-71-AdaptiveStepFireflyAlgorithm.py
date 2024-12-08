import numpy as np

class AdaptiveStepFireflyAlgorithm(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.step_size = 0.1  # Initial step size
    
    def __call__(self, func):
        def adaptive_step_update(x, x_new, f_x, f_new):
            if f_new < f_x:
                return x_new
            else:
                return x + self.step_size * (x_new - x)
        
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(indiv) for indiv in pop])
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        step_size = attraction(pop[i], pop[j])
                        new_candidate = levy_update(pop[i])
                        new_fitness = func(new_candidate)
                        pop[i] = adaptive_step_update(pop[i], new_candidate, fitness[i], new_fitness)
                        fitness[i] = new_fitness
            
            if np.random.rand() < 0.1:  # Probability of change
                self.pop_size = min(30, self.pop_size + 5)
                pop = np.vstack((pop, np.random.uniform(self.lb, self.ub, (5, self.dim)))
                fitness = np.append(fitness, [func(indiv) for indiv in pop[-5:]])

                if np.random.rand() < 0.2 and self.step_size > 0.01:  # Adaptive step size control
                    self.step_size *= 0.9
        
        return pop[np.argmin(fitness)]