import numpy as np

class HybridGADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 3 * dim)  # Ensure a diverse population
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.elite_fraction = 0.1
        self.eval_count = 0

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += len(population)
        return fitness

    def mutate(self, individual, population, best):
        # Differential Evolution-style mutation
        indices = np.random.choice(self.population_size, 3, replace=False)
        x1, x2, x3 = population[indices]
        mutant = x1 + self.mutation_factor * (x2 - x3)
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant):
        # Binomial Crossover
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, population, fitness, trial, trial_fitness, idx):
        if trial_fitness < fitness[idx]:
            population[idx] = trial
            fitness[idx] = trial_fitness

    def elite_local_search(self, best, func):
        # Simple local search around best individual
        epsilon = 0.1
        neighbors = np.clip(best + epsilon * np.random.uniform(-1, 1, (5, self.dim)), self.lower_bound, self.upper_bound)
        neighbor_fitness = self.evaluate_population(neighbors, func)
        best_idx = np.argmin(neighbor_fitness)
        return neighbors[best_idx], neighbor_fitness[best_idx]

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(population, func)
        
        while self.eval_count < self.budget:
            best_idx = np.argmin(fitness)
            best = population[best_idx]

            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                mutant = self.mutate(population[i], population, best)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                self.eval_count += 1
                self.select(population, fitness, trial, trial_fitness, i)

            # Local search on elite individuals
            num_elites = max(1, int(self.elite_fraction * self.population_size))
            elite_indices = np.argsort(fitness)[:num_elites]
            for idx in elite_indices:
                if self.eval_count >= self.budget:
                    break
                improved, improved_fitness = self.elite_local_search(population[idx], func)
                if improved_fitness < fitness[idx]:
                    population[idx] = improved
                    fitness[idx] = improved_fitness

        return population[np.argmin(fitness)]