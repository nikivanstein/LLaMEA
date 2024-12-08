import numpy as np

class DiversifiedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 8 * dim  # Adjusted population size
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.6  # Adjusted mutation factor
        self.crossover_rate = 0.8  # Adjusted crossover rate
        self.evaluations = 0
        self.adaptive_mutation_factor = 0.5  # Adjusted adaptive mutation factor
        self.success_rate = 0.08  # Adjusted success rate

    def __call__(self, func):
        def differential_evolution(population):
            new_population = []
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 5, replace=False)
                a, b, c, d = population[indices[0]], population[indices[1]], population[indices[2]], population[indices[3]]
                
                # Modify with probability
                if np.random.rand() < 0.25:
                    self.mutation_factor = np.random.uniform(0.3, self.adaptive_mutation_factor)
                mutant = np.clip(a + self.mutation_factor * (b - c) + 0.2 * (d - a), *self.bounds)
                
                trial = np.copy(population[i])
                crossover = np.random.uniform(size=self.dim) < self.crossover_rate
                trial[crossover] = mutant[crossover]
                
                if func(trial) < func(population[i]):
                    new_population.append(trial)
                    self.success_rate = min(1.0, self.success_rate + 0.03)
                else:
                    new_population.append(population[i])
                    self.success_rate = max(0.01, self.success_rate - 0.01)
                
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    return new_population
            return new_population
        
        def dynamic_local_search(individual):
            step_size = (self.bounds[1] - self.bounds[0]) * 0.002
            best_local = np.copy(individual)
            best_val = func(best_local)
            
            for _ in range(20):
                perturbation = np.random.normal(0, step_size, self.dim)
                candidate = np.clip(best_local + perturbation, *self.bounds)
                
                candidate_val = func(candidate)
                self.evaluations += 1
                if candidate_val < best_val:
                    best_local, best_val = candidate, candidate_val
                
                if self.evaluations >= self.budget:
                    break
            
            return best_local

        population = np.random.uniform(*self.bounds, (self.population_size, self.dim))
        
        while self.evaluations < self.budget:
            population = differential_evolution(population)
            
            # Modify with probability
            if self.evaluations < self.budget * (0.7 if np.random.rand() < 0.25 else 0.8):
                population = sorted(population, key=func)
                best_count = max(2, self.population_size // 14)
                for i in range(min(best_count, len(population))):
                    # Modify with probability
                    if np.random.rand() < 0.25:
                        population[i] = dynamic_local_search(population[i])
                    if self.evaluations >= self.budget:
                        break

        best_individual = min(population, key=func)
        return best_individual