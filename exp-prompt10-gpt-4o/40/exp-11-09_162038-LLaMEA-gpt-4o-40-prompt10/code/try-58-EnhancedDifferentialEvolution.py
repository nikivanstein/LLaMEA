import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 8)  # Increased initial population for diverse exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.5  # Adjusted for balance between exploration and exploitation
        self.CR_base = 0.9  # Increased crossover probability
        self.adaptation_rate = 0.05  # Increased adaptive change rate
        self.local_search_intensity = 0.15  # Increased intensity of local search
        self.mutation_strategy = 'rand'  # Primary strategy with opposition-based learning

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        # Track the best solution found
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Adaptive F and CR
                F = np.abs(self.F_base + self.adaptation_rate * np.random.randn())
                CR = np.clip(self.CR_base + self.adaptation_rate * np.random.randn(), 0.0, 1.0)

                if np.random.rand() < 0.5:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(self.population_size, 2, replace=False)
                    a, b = population[indices]
                    mutant = np.clip(best_individual + F * (a - b), self.lower_bound, self.upper_bound)

                # Opposition-based learning
                opposite = self.lower_bound + self.upper_bound - mutant
                opposite = np.clip(opposite, self.lower_bound, self.upper_bound)

                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                trial_fitness = func(trial)
                eval_count += 1

                # Evaluate opposite individual
                opposite_fitness = func(opposite)
                eval_count += 1

                # Selection
                if trial_fitness < fitness[i] and trial_fitness <= opposite_fitness:
                    population[i] = trial
                    fitness[i] = trial_fitness
                elif opposite_fitness < fitness[i]:
                    population[i] = opposite
                    fitness[i] = opposite_fitness

                # Update best if needed
                if fitness[i] < best_fitness:
                    best_individual = population[i]
                    best_fitness = fitness[i]

                if eval_count >= self.budget:
                    break

            # Enhanced local search around the best individual
            neighborhood_size = int(self.local_search_intensity * self.population_size)
            local_neighbors = best_individual + np.random.uniform(-0.1, 0.1, (neighborhood_size, self.dim))
            local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
            local_fitness = np.array([func(ind) for ind in local_neighbors])
            eval_count += len(local_neighbors)

            if np.min(local_fitness) < best_fitness:
                best_local_index = np.argmin(local_fitness)
                best_individual = local_neighbors[best_local_index]
                best_fitness = local_fitness[best_local_index]

            # Self-adaptive population management
            if eval_count < self.budget * 0.5 and np.random.rand() < 0.1:
                new_indices = np.random.choice(self.population_size, size=self.population_size // 2, replace=False)
                new_population = np.random.uniform(self.lower_bound, self.upper_bound, (len(new_indices), self.dim))
                for idx, new_ind in zip(new_indices, new_population):
                    population[idx] = new_ind
                    fitness[idx] = func(new_ind)
                    eval_count += 1

            population[0] = best_individual
            fitness[0] = best_fitness

        return best_individual