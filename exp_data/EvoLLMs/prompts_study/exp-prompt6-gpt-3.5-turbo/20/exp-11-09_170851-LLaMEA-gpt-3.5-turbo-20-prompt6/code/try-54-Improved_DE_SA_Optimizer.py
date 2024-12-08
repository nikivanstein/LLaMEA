import numpy as np

class Improved_DE_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 15
        self.cr = 0.7
        self.temp = 1.0
        self.alpha = 0.95
        self.diversity_threshold = 0.05
        self.mutation_factors = np.ones(self.pop_size) * 0.5  # Initialize mutation factors dynamically
    
    def __call__(self, func):
        def mutate(x, pop, i, f):
            candidates = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
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
                new_sol = mutate(pop[i], pop, i, self.mutation_factors[i])
                new_cost = func(new_sol)
                
                if new_cost < costs[i]:
                    pop[i] = new_sol
                    costs[i] = new_cost
                    if new_cost < costs[best_idx]:
                        best_idx = i
                        best_sol = new_sol
                    self.mutation_factors[i] = max(0.1, min(0.8, self.mutation_factors[i] + 0.2 * (costs[i] - new_cost)))  # Update mutation factor dynamically
                elif np.random.rand() < self.cr:
                    pop[i] = new_sol
                    costs[i] = new_cost
                    self.mutation_factors[i] = max(0.1, min(0.8, self.mutation_factors[i] - 0.1))
                    
            if np.std(pop) < self.diversity_threshold:
                pop = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
                
            new_temp = self.alpha * self.temp
            if new_temp > 0.0:
                p = acceptance_probability(costs[best_idx], func(best_sol), self.temp)
                if np.random.rand() < p:
                    self.temp = new_temp
            self.pop_size = 15
            
        return best_sol