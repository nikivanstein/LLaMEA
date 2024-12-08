import numpy as np

class AdaptiveHybridDECS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.F = 0.7  # Mutation factor for Differential Evolution
        self.CR = 0.9  # Crossover probability for Differential Evolution
        self.pa = 0.3  # Discovery rate of alien eggs/solutions in Cuckoo Search
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness_values = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            # Differential Evolution (DE) Update
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                evaluations += 1
                
                if f_trial < fitness_values[i]:
                    fitness_values[i] = f_trial
                    population[i] = trial
                    
                    if f_trial < fitness_values[best_idx]:
                        best_idx = i
                        best_solution = population[best_idx]

                if evaluations >= self.budget:
                    break

            # Enhanced Cuckoo Search (CS) inspired update
            for i in range(self.pop_size):
                if np.random.rand() < self.pa:
                    step_size = np.random.standard_cauchy(size=self.dim)
                    new_solution = population[i] + step_size * (population[i] - best_solution)
                    new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                    
                    f_new = func(new_solution)
                    evaluations += 1
                    
                    if f_new < fitness_values[i]:
                        fitness_values[i] = f_new
                        population[i] = new_solution
                        
                        if f_new < fitness_values[best_idx]:
                            best_idx = i
                            best_solution = population[best_idx]

                if evaluations >= self.budget:
                    break
        
        return best_solution