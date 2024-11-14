import numpy as np

class NADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(15, 5 * dim)
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.init_std = np.std([self.lower_bound, self.upper_bound])
    
    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf
        evaluations = 0

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation step
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover step
                crossover = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover, mutant, population[i])
                
                # Evaluation
                trial_fitness = func(trial)
                evaluations += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            # Adapt mutation factor and crossover probability
            diversity = np.mean(np.std(population, axis=0)) / self.init_std
            self.mutation_factor = 0.5 + 0.3 * np.random.rand() * diversity
            self.crossover_prob = 0.6 + 0.2 * np.random.rand() * (1 - diversity)

        return best_solution