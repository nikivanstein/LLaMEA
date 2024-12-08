import numpy as np

class HybridDifferentialEvolutionSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))
        self.F = 0.8  # differential weight
        self.CR = 0.9  # crossover probability
        self.initial_temperature = 1.0
        self.cooling_rate = 0.99

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        temperature = self.initial_temperature

        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Acceptance
                trial_fitness = func(trial)
                num_evaluations += 1
                
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            temperature *= self.cooling_rate

        return best_solution, best_fitness