import numpy as np

class SwarmBasedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(10, int(budget / (5 * dim)))  # heuristic for population size
        self.mutation_factor = 0.8  # F in DE
        self.crossover_rate = 0.9  # CR in DE

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size

        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                # Select three random distinct indices different from i
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Mutation and Crossover
                mutant = population[a] + self.mutation_factor * (population[b] - population[c])
                mutant = np.clip(mutant, self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                num_evaluations += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                
                if num_evaluations >= self.budget:
                    break
            
            population = new_population

        return best_solution, best_fitness