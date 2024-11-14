import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(60, self.budget // 8)  # Increased population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.7  # Improved differential weight for exploration-exploitation
        self.CR_base = 0.9  # Adjusted crossover probability
        self.adaptation_rate = 0.05  # Increased adaptive change rate
        self.local_search_intensity = 0.15  # Enhanced local search intensity
        self.cooling_rate = 0.99  # Adaptive cooling schedule
        self.mutation_strategy = 'rand-to-best'  # New strategy for mutation

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        # Track the best solution found
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]
        current_CR = self.CR_base

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Adaptive F and CR with cooling
                F = self.F_base * (self.cooling_rate ** (eval_count / self.budget))
                current_CR = max(0.1, current_CR * (1 - self.adaptation_rate))

                # Enhanced mutation strategy
                target = population[i]
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(target + F * (best_individual - target) + F * (a - b), self.lower_bound, self.upper_bound)

                # Self-repairing crossover
                trial = np.where(np.random.rand(self.dim) < current_CR, mutant, target)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Evaluate trial individual with stochastic tunneling
                trial_fitness = func(trial) + 0.01 * (np.random.rand() - 0.5)
                eval_count += 1

                # Selection and elitism
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            # Cooperative local search around the best individual
            neighborhood_size = int(self.local_search_intensity * self.population_size)
            local_neighbors = best_individual + np.random.uniform(-0.05, 0.05, (neighborhood_size, self.dim))
            local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
            local_fitness = np.array([func(ind) for ind in local_neighbors])
            eval_count += len(local_neighbors)

            # Update best if any local neighbor is better
            if np.min(local_fitness) < best_fitness:
                best_local_index = np.argmin(local_fitness)
                best_individual = local_neighbors[best_local_index]
                best_fitness = local_fitness[best_local_index]

            population[0] = best_individual
            fitness[0] = best_fitness

        # Return best found solution
        return best_individual