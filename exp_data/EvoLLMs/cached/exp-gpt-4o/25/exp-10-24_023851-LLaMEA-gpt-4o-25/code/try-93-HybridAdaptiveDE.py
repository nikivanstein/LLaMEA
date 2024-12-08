import numpy as np

class HybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 8 * dim  # Adjusted population size for better exploration
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8  # Modified mutation factor for enhanced search
        self.crossover_rate = 0.85  # Balanced crossover rate
        self.evaluations = 0
        self.adaptive_mutation_factor = 0.6
        self.success_rate = 0.15  # Modified higher initial success rate

    def __call__(self, func):
        def differential_evolution(population):
            new_population = []
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 5, replace=False)
                a, b, c, d = population[indices[0]], population[indices[1]], population[indices[2]], population[indices[3]]

                if np.random.rand() < self.success_rate:
                    self.mutation_factor = np.random.uniform(0.4, self.adaptive_mutation_factor)

                mutant = np.clip(a + self.mutation_factor * (b - c), *self.bounds)  # Standard DE strategy
                
                trial = np.copy(population[i])
                crossover = np.random.uniform(size=self.dim) < self.crossover_rate
                trial[crossover] = mutant[crossover]

                if func(trial) < func(population[i]):
                    new_population.append(trial)
                    self.success_rate = min(1.0, self.success_rate + 0.03)  # Adjusted success rate increment
                else:
                    new_population.append(population[i])
                    self.success_rate = max(0.05, self.success_rate - 0.01)  # Adjusted success rate decrement

                self.evaluations += 1
                if self.evaluations >= self.budget:
                    return new_population
            return new_population
        
        def adaptive_local_search(individual):
            step_size = (self.bounds[1] - self.bounds[0]) * 0.002  # Adjusted step size for fine-tuning
            best_local = np.copy(individual)
            best_val = func(best_local)

            for _ in range(20):  # Increased iterations for thorough local search
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

            if self.evaluations < self.budget * 0.5:  # Adjusted exploration phase
                population = sorted(population, key=func)
                best_count = max(3, self.population_size // 8)  # Adjusted number of best individuals
                for i in range(min(best_count, len(population))):
                    population[i] = adaptive_local_search(population[i])
                    if self.evaluations >= self.budget:
                        break

        best_individual = min(population, key=func)
        return best_individual