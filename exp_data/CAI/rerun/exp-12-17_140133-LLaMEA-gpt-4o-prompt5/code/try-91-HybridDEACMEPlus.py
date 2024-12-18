import numpy as np

class HybridDEACMEPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(4, 10 + int(self.dim ** 0.5))
        self.CR = 0.9
        self.F = np.random.uniform(0.5, 1.0)  # Adaptive F for enhanced balance
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evals = 0
        
    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf
        
        self.fitness = np.array([func(ind) for ind in self.population])
        self.evals += self.population_size
        
        while self.evals < self.budget:
            f_scale = np.random.uniform(0.6, 1.1)  # Narrowed f_scale for stability
            elite = self.population[np.argmin(self.fitness)]
            avg_fitness = np.mean(self.fitness)
            adaptive_F = self.F * (1 - np.std(self.fitness) / avg_fitness)  # Adaptive F adjustment
            
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + f_scale * (b - c) + (adaptive_F / 2) * (elite - self.population[i]), self.lower_bound, self.upper_bound)
                
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, self.population[i])
                
                trial_fitness = func(trial)
                self.evals += 1
                
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            center = np.mean(self.population, axis=0)
            covariance = np.cov(self.population, rowvar=False) + np.eye(self.dim) * 1e-5
            new_population = np.random.multivariate_normal(center, covariance, size=self.population_size)
            new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
            
            for j in range(self.population_size):
                new_fitness = func(new_population[j])
                self.evals += 1
                if new_fitness < self.fitness[j]:
                    self.population[j] = new_population[j]
                    self.fitness[j] = new_fitness
                    if new_fitness < best_fitness:
                        best_solution = new_population[j]
                        best_fitness = new_fitness

                if self.evals >= self.budget:
                    break

        return best_solution