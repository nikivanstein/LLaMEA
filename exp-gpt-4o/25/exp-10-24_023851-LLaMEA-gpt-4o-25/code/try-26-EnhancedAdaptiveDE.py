import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 8 * dim
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.7
        self.crossover_rate = 0.85
        self.evaluations = 0
        self.adaptive_mutation_factor = 0.6
        self.diversity_threshold = 0.1
    
    def __call__(self, func):
        def differential_evolution(population):
            new_population = []
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 4, replace=False)
                a, b, c = population[indices[0]], population[indices[1]], population[indices[2]]
                
                self.mutation_factor = self.adaptive_mutation_factor if np.random.rand() < 0.5 else self.mutation_factor
                
                mutant = np.clip(a + self.mutation_factor * (b - c), *self.bounds)
                
                trial = np.copy(population[i])
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial[crossover] = mutant[crossover]
                
                if func(trial) < func(population[i]):
                    new_population.append(trial)
                else:
                    new_population.append(population[i])
                
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    return new_population
            return new_population
        
        def probabilistic_neighborhood_search(individual):
            step_size = (self.bounds[1] - self.bounds[0]) * 0.007
            best_local = np.copy(individual)
            best_val = func(best_local)
            
            for _ in range(10):
                perturbation = np.random.uniform(-step_size, step_size, self.dim)
                candidate = np.clip(best_local + perturbation, *self.bounds)
                
                candidate_val = func(candidate)
                self.evaluations += 1
                if candidate_val < best_val:
                    best_local, best_val = candidate, candidate_val
                
                if self.evaluations >= self.budget:
                    break
            
            return best_local
        
        def preserve_diversity(population):
            diversity = np.std(population, axis=0)
            if np.any(diversity < self.diversity_threshold):
                additional_population = np.random.uniform(*self.bounds, (self.population_size // 5, self.dim))
                population = np.vstack((population, additional_population))
                population = population[:self.population_size]
            return population

        population = np.random.uniform(*self.bounds, (self.population_size, self.dim))
        
        while self.evaluations < self.budget:
            population = differential_evolution(population)
            population = preserve_diversity(population)
            
            if self.evaluations < self.budget * 0.6:
                population = sorted(population, key=func)
                best_count = max(3, self.population_size // 18)
                for i in range(min(best_count, len(population))):
                    population[i] = probabilistic_neighborhood_search(population[i])
                    if self.evaluations >= self.budget:
                        break

        best_individual = min(population, key=func)
        return best_individual