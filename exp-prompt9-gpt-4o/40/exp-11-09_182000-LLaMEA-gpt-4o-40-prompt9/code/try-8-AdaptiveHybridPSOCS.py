import numpy as np

class AdaptiveHybridPSOCS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.pa = 0.3  # Probability of alien egg discovery in Cuckoo Search

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_sol = population[best_idx]
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            # Differential Evolution (DE) Mutation and Crossover
            for i in range(self.pop_size):
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x0, x1, x2 = population[idxs]
                mutant = np.clip(x0 + self.F * (x1 - x2), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate trial vector
                f_val = func(trial)
                evaluations += 1
                
                # Selection
                if f_val < fitness[i]:
                    fitness[i] = f_val
                    population[i] = trial
                    
                    # Update best solution
                    if f_val < fitness[best_idx]:
                        best_idx = i
                        best_sol = trial

                if evaluations >= self.budget:
                    break

            # Cuckoo Search (CS) inspired update
            for i in range(self.pop_size):
                if np.random.rand() < self.pa:
                    step_size = np.random.standard_cauchy(size=self.dim)
                    new_solution = population[i] + step_size * (population[i] - best_sol)
                    new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                    
                    f_new = func(new_solution)
                    evaluations += 1
                    
                    if f_new < fitness[i]:
                        fitness[i] = f_new
                        population[i] = new_solution
                        
                        if f_new < fitness[best_idx]:
                            best_idx = i
                            best_sol = new_solution

                if evaluations >= self.budget:
                    break

        return best_sol