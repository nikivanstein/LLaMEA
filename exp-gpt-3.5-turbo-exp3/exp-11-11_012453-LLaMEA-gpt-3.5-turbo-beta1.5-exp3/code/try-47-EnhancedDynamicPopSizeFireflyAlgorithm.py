import numpy as np

class EnhancedDynamicPopSizeFireflyAlgorithm(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_prob = 0.1  # Probability of applying local search
        
    def local_search(self, x, func):
        best_x = np.copy(x)
        best_fitness = func(x)
        for _ in range(5):  # Perform local search for 5 iterations
            new_x = np.clip(x + 0.1 * np.random.randn(self.dim), self.lb, self.ub)
            new_fitness = func(new_x)
            if new_fitness < best_fitness:
                best_x = np.copy(new_x)
                best_fitness = new_fitness
        return best_x
    
    def __call__(self, func):
        # Existing code for DynamicPopSizeFireflyAlgorithm
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        step_size = attraction(pop[i], pop[j])
                        pop[i] = levy_update(pop[i]) if step_size > np.random.rand() else pop[i]
                        fitness[i] = func(pop[i])
                        
                        # Integrate local search
                        if np.random.rand() < self.local_search_prob:
                            pop[i] = self.local_search(pop[i], func)
                            fitness[i] = func(pop[i])
            
            # Remaining code for DynamicPopSizeFireflyAlgorithm
        
        return pop[np.argmin(fitness)]