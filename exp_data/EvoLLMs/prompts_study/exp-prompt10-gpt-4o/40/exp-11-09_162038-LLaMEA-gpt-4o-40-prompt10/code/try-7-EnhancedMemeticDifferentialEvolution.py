import numpy as np

class EnhancedMemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(60, self.budget // 8)  # Increased population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_min = 0.5  # Minimum differential weight
        self.F_max = 0.9  # Maximum differential weight
        self.CR = 0.8  # Crossover probability, slightly reduced to promote diversity

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                adaptive_F = self.F_min + np.random.rand() * (self.F_max - self.F_min)
                mutant = np.clip(a + adaptive_F * (b - c), self.lower_bound, self.upper_bound)
                
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                
                trial_fitness = func(trial)
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if eval_count >= self.budget:
                    break

            best_index = np.argmin(fitness)
            best_individual = population[best_index]
            best_fitness = fitness[best_index]
            
            local_search_radius = np.linspace(0.05, 0.1, num=10)  # Dynamic local search radius
            for radius in local_search_radius:
                if eval_count >= self.budget:
                    break
                local_neighbors = best_individual + np.random.uniform(-radius, radius, (5, self.dim))
                local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
                local_fitness = np.array([func(ind) for ind in local_neighbors])
                eval_count += len(local_neighbors)
                
                if np.min(local_fitness) < best_fitness:
                    best_index = np.argmin(local_fitness)
                    best_individual = local_neighbors[best_index]
                    best_fitness = local_fitness[best_index]
            
            population[0] = best_individual
            fitness[0] = best_fitness

        best_index = np.argmin(fitness)
        return population[best_index]