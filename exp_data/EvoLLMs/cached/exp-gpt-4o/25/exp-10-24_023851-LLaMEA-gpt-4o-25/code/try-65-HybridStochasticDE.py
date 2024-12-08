import numpy as np

class HybridStochasticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim  # Adjusted population size
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8  # Adjusted mutation factor
        self.crossover_rate = 0.9  # Adjusted crossover rate
        self.evaluations = 0
        self.adaptive_mutation_factor = 0.5  # Adjusted adaptive mutation factor
        self.success_rate = 0.15  # Adjusted success rate
        self.gradient_step_size = 0.005  # New parameter for gradient-based updates

    def __call__(self, func):
        def differential_evolution(population):
            new_population = []
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 5, replace=False)
                a, b, c, d = population[indices[0]], population[indices[1]], population[indices[2]], population[indices[3]]
                
                if np.random.rand() < self.success_rate:
                    self.mutation_factor = np.random.uniform(0.3, self.adaptive_mutation_factor)  # Adjust range

                mutant = np.clip(a + self.mutation_factor * (b - c) + 0.15 * (d - b), *self.bounds)  # Adjust contribution
                
                trial = np.copy(population[i])
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial[crossover] = mutant[crossover]
                
                if func(trial) < func(population[i]):
                    new_population.append(trial)
                    self.success_rate = min(1.0, self.success_rate + 0.03)  # Adjust success rate increment
                else:
                    new_population.append(population[i])
                    self.success_rate = max(0.02, self.success_rate - 0.01)  # Adjust success rate decrement
                
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    return new_population
            return new_population
        
        def stochastic_gradient_update(individual):
            gradient = np.random.normal(0, self.gradient_step_size, self.dim)  # Gradient-based update
            candidate = np.clip(individual - gradient, *self.bounds)
            candidate_val = func(candidate)
            self.evaluations += 1
            if candidate_val < func(individual):
                return candidate
            return individual

        population = np.random.uniform(*self.bounds, (self.population_size, self.dim))
        
        while self.evaluations < self.budget:
            population = differential_evolution(population)
            
            if self.evaluations < self.budget * 0.8:  # Adjust exploration-exploitation balance
                population = sorted(population, key=func)
                best_count = max(4, self.population_size // 10)  # Adjust number of best individuals
                for i in range(min(best_count, len(population))):
                    population[i] = stochastic_gradient_update(population[i])
                    if self.evaluations >= self.budget:
                        break

        best_individual = min(population, key=func)
        return best_individual