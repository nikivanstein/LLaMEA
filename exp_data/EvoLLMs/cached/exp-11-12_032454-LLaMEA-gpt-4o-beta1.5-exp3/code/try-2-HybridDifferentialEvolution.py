import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(10, int(budget / (8 * dim)))  # heuristic for population size
        self.mutation_factor = 0.5  # initial mutation factor
        self.crossover_rate = 0.9  # initial crossover rate
        self.stochastic_perturbation_rate = 0.1  # fraction of individuals with stochastic perturbation
        self.adaptive_rate = 0.1  # rate at which parameters adapt

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        while num_evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                # Mutation: Differential Evolution strategy
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant_vector = x1 + self.mutation_factor * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, self.lb, self.ub)

                # Crossover: Binomial crossover
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])

                # Stochastic Perturbation
                if np.random.rand() < self.stochastic_perturbation_rate:
                    perturbation = np.random.normal(0, 0.1, self.dim)
                    trial_vector = np.clip(trial_vector + perturbation, self.lb, self.ub)

                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                num_evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial_vector
                        best_fitness = trial_fitness
                else:
                    new_population.append(population[i])

            population = np.array(new_population)

            # Adaptive Parameter Control
            self.mutation_factor = max(0.4, self.mutation_factor - self.adaptive_rate * (1.0 - num_evaluations / self.budget))
            self.crossover_rate = min(0.95, self.crossover_rate + self.adaptive_rate * (num_evaluations / self.budget))

        return best_solution, best_fitness