import numpy as np

class HybridDifferentialEvolutionSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (5 * dim)))
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.initial_temperature = 100
        self.cooling_rate = 0.99

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best_position = np.copy(population[best_idx])
        best_fitness = fitness[best_idx]
        
        temperature = self.initial_temperature
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Differential Evolution Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lb, self.ub)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.crossover_probability
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Simulated Annealing Acceptance
                trial_fitness = func(trial)
                num_evaluations += 1
                
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_position = np.copy(trial)
                        best_fitness = trial_fitness
            
            # Update temperature with cooling
            temperature *= self.cooling_rate
        
        return best_position, best_fitness