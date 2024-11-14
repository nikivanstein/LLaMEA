import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = max(4, 5 * dim)  # Reduced initial size for faster convergence
        self.mutation_factor = 0.8
        self.initial_crossover_rate = 0.9
        self.dynamic_population = True

    def __call__(self, func):
        np.random.seed(0)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evals = self.initial_population_size
        crossover_rate = self.initial_crossover_rate

        while evals < self.budget:
            new_population = []
            for i in range(len(population)):
                idxs = list(range(len(population)))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, population[i])

                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    new_population.append(trial)
                else:
                    new_population.append(population[i])

                if evals >= self.budget:
                    break

            # Dynamic crossover rate adjustment
            crossover_rate = self.initial_crossover_rate * (1 - evals / self.budget)
            if self.dynamic_population and len(new_population) > 2:
                # Reduce population size as evaluations increase to focus search
                population = np.array(new_population[::2])
            else:
                population = np.array(new_population)

        best_index = np.argmin(fitness)
        return population[best_index]