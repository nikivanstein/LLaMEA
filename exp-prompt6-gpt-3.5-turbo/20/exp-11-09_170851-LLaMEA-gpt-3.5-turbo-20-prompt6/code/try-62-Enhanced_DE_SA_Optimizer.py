import numpy as np

class Enhanced_DE_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 15
        self.cr = 0.7  
        self.temp = 1.0  
        self.alpha = 0.95  
        self.diversity_threshold = 0.05  
    
    def __call__(self, func):
        def mutate(x, pop, i, f):
            candidates = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            f = 0.5 if np.random.rand() < 0.5 else np.clip(f + 0.2 * np.random.randn(), 0.1, 0.8)
            mutated = np.clip(a + f * (b - c), -5.0, 5.0)
            return mutated
        
        def acceptance_probability(curr_cost, new_cost, temp):
            if new_cost < curr_cost:
                return 1.0
            return np.exp((curr_cost - new_cost) / temp)
        
        pop = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        costs = [func(ind) for ind in pop]
        best_idx = np.argmin(costs)
        best_sol = pop[best_idx]
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                new_sol = mutate(pop[i], pop, i, 0.5)
                new_cost = func(new_sol)
                
                if new_cost < costs[i]:
                    pop[i] = new_sol
                    costs[i] = new_cost
                    if new_cost < costs[best_idx]:
                        best_idx = i
                        best_sol = new_sol
                    
            new_temp = self.alpha * self.temp
            if new_temp > 0.0:
                p = acceptance_probability(costs[best_idx], func(best_sol), self.temp)
                if np.random.rand() < p:
                    self.temp = new_temp

        return best_sol