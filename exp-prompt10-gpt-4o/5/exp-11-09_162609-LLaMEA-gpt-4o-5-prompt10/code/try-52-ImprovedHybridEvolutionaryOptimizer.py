import numpy as np

class ImprovedHybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(10 * np.log(self.dim))
        self.bounds = (-5.0, 5.0)
        self.scale_factor = 0.8
        self.cross_prob = 0.9
        self.adaptation_rate = 0.05
        self.elitism_rate = 0.2  # New parameter for elitism

    def __call__(self, func):
        # Initialize population within bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        func_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        while func_evals < self.budget:
            # Ensure a portion of the best solutions are retained (elitism)
            num_elites = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(fitness)[:num_elites]
            elites = population[elite_indices]

            for i in range(self.population_size):
                if func_evals >= self.budget:
                    break

                # Mutation: choose three random indices different from i
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Dynamic Adaptive Differential Mutation
                adapt_factor = self.adaptation_rate * (np.random.rand() - 0.5)
                mutant = a + (self.scale_factor + adapt_factor) * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Enhanced Crossover with dynamic probability
                trial = np.copy(population[i])
                crossover = np.random.rand(self.dim) < (self.cross_prob + adapt_factor * np.random.uniform(0.9, 1.1))
                trial[crossover] = mutant[crossover]

                # Selection
                trial_fitness = func(trial)
                func_evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    # Update global best
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            # Replace worst individuals with elites to preserve good solutions
            worst_indices = np.argsort(fitness)[-num_elites:]
            population[worst_indices] = elites
            fitness[worst_indices] = [func(ind) for ind in elites]
            func_evals += num_elites

            # Self-adaptive parameter tuning with more granular adjustments
            self.scale_factor = np.random.uniform(0.65, 0.85)
            self.cross_prob = np.random.uniform(0.88, 1.0)

        # Return the best found solution
        return best_solution, best_fitness