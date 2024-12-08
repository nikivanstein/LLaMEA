import numpy as np

class Enhanced_DE_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.cr = 0.7  # Crossover rate for DE
        self.temp = 1.0  # Initial temperature for SA
        self.alpha = 0.95  # Cooling rate for SA
        
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
        
        f_vals = np.ones(self.pop_size) * 0.5  # Initial mutation factor
        elite_idx = np.argsort(costs)[:int(self.pop_size * 0.3)]  # Select top 30% as elite
        for _ in range(self.budget):
            for i in range(self.pop_size):
                new_sol = mutate(pop[i], pop, i, f_vals[i])
                new_cost = func(new_sol)
                
                if new_cost < costs[i]:
                    pop[i] = new_sol
                    costs[i] = new_cost
                    if new_cost < costs[best_idx]:
                        best_idx = i
                        best_sol = new_sol
                    f_vals[i] = max(0.1, min(0.8, f_vals[i] + 0.1))  # Dynamic adjustment of mutation factor
                elif np.random.rand() < self.cr:
                    pop[i] = new_sol
                    costs[i] = new_cost
                    f_vals[i] = max(0.1, min(0.8, f_vals[i] - 0.05))  # Dynamic adjustment of mutation factor
                    
            # Simulated Annealing
            new_temp = self.alpha * self.temp
            if new_temp > 0.0:
                p = acceptance_probability(costs[best_idx], func(best_sol), self.temp)
                if np.random.rand() < p:
                    self.temp = new_temp
            
        return best_sol