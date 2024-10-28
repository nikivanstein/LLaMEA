import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Increased population size for diversity
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8  # Further increased mutation factor
        self.crossover_rate = 0.85  # Slightly decreased crossover rate for reduced premature convergence
        self.evaluations = 0
        self.adaptive_mutation_factor = 0.4
        self.success_rate = 0.15  # Increased initial success rate for faster adaptation

    def __call__(self, func):
        def differential_evolution(population):
            new_population = []
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 5, replace=False)
                a, b, c, d = population[indices[0]], population[indices[1]], population[indices[2]], population[indices[3]]

                if np.random.rand() < self.success_rate:
                    self.mutation_factor = np.random.uniform(0.2, self.adaptive_mutation_factor)  

                mutant = np.clip(a + self.mutation_factor * (b - c) + 0.3 * (d - a), *self.bounds)  # Further increased contribution factor
                
                trial = np.copy(population[i])
                crossover = np.random.uniform(size=self.dim) < self.crossover_rate  
                trial[crossover] = mutant[crossover]

                if func(trial) < func(population[i]):
                    new_population.append(trial)
                    self.success_rate = min(1.0, self.success_rate + 0.05)  # Further increased success rate increment
                else:
                    new_population.append(population[i])
                    self.success_rate = max(0.01, self.success_rate - 0.03)  # Further decreased success rate decrement

                self.evaluations += 1
                if self.evaluations >= self.budget:
                    return new_population
            return new_population
        
        def enhanced_local_search(individual):
            step_size = (self.bounds[1] - self.bounds[0]) * 0.002  # Reduced step size for nuanced local search
            best_local = np.copy(individual)
            best_val = func(best_local)

            for _ in range(20):  # Increased iterations for thorough exploitation
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

            if self.evaluations < self.budget * 0.5:  # Adjusted exploration phase ratio
                population = sorted(population, key=func)
                best_count = max(3, self.population_size // 8)  # Slightly increased number of best individuals
                for i in range(min(best_count, len(population))):
                    population[i] = enhanced_local_search(population[i])
                    if self.evaluations >= self.budget:
                        break

        best_individual = min(population, key=func)
        return best_individual