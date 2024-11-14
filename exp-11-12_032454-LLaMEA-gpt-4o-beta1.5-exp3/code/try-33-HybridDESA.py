import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.temperature = 1.0  # Initial temperature for simulated annealing

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Differential Evolution step
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[idxs]
                mutant = a + self.f * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)
                
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate trial solution
                trial_fitness = func(trial)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                else:
                    # Simulated Annealing acceptance criterion
                    acceptance_probability = np.exp((fitness[i] - trial_fitness) / self.temperature)
                    if np.random.rand() < acceptance_probability:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        
                # Cooling schedule
                self.temperature *= 0.99

        return best_solution, best_fitness