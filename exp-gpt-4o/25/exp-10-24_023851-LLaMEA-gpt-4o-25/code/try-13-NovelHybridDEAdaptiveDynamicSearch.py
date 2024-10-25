import numpy as np

class NovelHybridDEAdaptiveDynamicSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim  # Adjusted population size
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.7  # Modified mutation factor
        self.crossover_rate = 0.85  # Modified crossover rate
        self.evaluations = 0
        self.adaptive_mutation_factor = 0.6  # Enhanced adaptive mutation factor
    
    def __call__(self, func):
        def differential_evolution(population):
            new_population = []
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 4, replace=False)
                a, b, c = population[indices[0]], population[indices[1]], population[indices[2]]
                
                # Strategic Adaptive Mutation
                self.mutation_factor = self.adaptive_mutation_factor if np.random.rand() < 0.6 else self.mutation_factor
                
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
        
        def dynamic_local_search(individual):
            step_size = (self.bounds[1] - self.bounds[0]) * 0.004  # Adjusted step size
            best_local = np.copy(individual)
            best_val = func(best_local)
            
            for _ in range(12):  # Enhanced the number of local search steps
                perturbation = np.random.uniform(-step_size, step_size, self.dim)
                candidate = np.clip(best_local + perturbation, *self.bounds)
                
                candidate_val = func(candidate)
                self.evaluations += 1
                if candidate_val < best_val:
                    best_local, best_val = candidate, candidate_val
                
                if self.evaluations >= self.budget:
                    break
            
            return best_local

        # Initialize random population
        population = np.random.uniform(*self.bounds, (self.population_size, self.dim))
        
        while self.evaluations < self.budget:
            # Perform Differential Evolution
            population = differential_evolution(population)
            
            # Apply Dynamic Local Search to the best individuals
            if self.evaluations < self.budget / 2:
                population = sorted(population, key=func)
                best_count = max(3, self.population_size // 15)  # Enhanced best individuals count
                for i in range(min(best_count, len(population))):
                    population[i] = dynamic_local_search(population[i])
                    if self.evaluations >= self.budget:
                        break

        # Return the best solution found
        best_individual = min(population, key=func)
        return best_individual