import numpy as np

class DynamicMultiStrategyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = population_size
        mutation_strategies = [self.de_rand_1, self.de_best_1]
        adaptive_params = [0.8, 0.9]  # Initial values for mutation_factor and crossover_rate

        def local_search(individual):
            adapted_step_size = np.clip(0.02 * np.exp(-0.005 * evaluations / self.budget), 0.001, 0.05)
            candidate = individual + np.random.normal(0, adapted_step_size, self.dim)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            return candidate

        while evaluations < self.budget:
            progress_ratio = evaluations / self.budget
            mutation_factor = adaptive_params[0] - 0.3 * (progress_ratio ** 0.5)
            crossover_rate = adaptive_params[1] * (1 - progress_ratio)

            # Select and apply mutation strategies
            strategy_choice = np.random.choice(mutation_strategies)
            new_population, new_fitness = strategy_choice(population, fitness, mutation_factor, crossover_rate, func)

            # Update population and fitness
            for i in range(population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    if new_fitness[i] < best_fitness:
                        best_solution = new_population[i]
                        best_fitness = new_fitness[i]

            evaluations += population_size

            # Adaptive local search
            local_search_indices = np.random.choice(population_size, int(0.1 * population_size), replace=False)
            for idx in local_search_indices:
                candidate = local_search(population[idx])
                candidate_fitness = func(candidate)
                evaluations += 1
                if candidate_fitness < fitness[idx]:
                    population[idx] = candidate
                    fitness[idx] = candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_solution = candidate
                        best_fitness = candidate_fitness

            # Adjust adaptive parameters based on performance
            adaptive_params[0] = mutation_factor * 0.9 + 0.1 * adaptive_params[0]
            adaptive_params[1] = crossover_rate * 0.9 + 0.1 * adaptive_params[1]

            if evaluations >= self.budget:
                break

        return best_solution

    def de_rand_1(self, population, fitness, mutation_factor, crossover_rate, func):
        population_size = len(population)
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        for i in range(population_size):
            indices = [idx for idx in range(population_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
            trial = [mutant[j] if np.random.rand() < crossover_rate else population[i][j] for j in range(self.dim)]
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
        return new_population, new_fitness

    def de_best_1(self, population, fitness, mutation_factor, crossover_rate, func):
        population_size = len(population)
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        best_idx = np.argmin(fitness)
        for i in range(population_size):
            indices = [idx for idx in range(population_size) if idx != i]
            a, b = population[np.random.choice(indices, 2, replace=False)]
            mutant = np.clip(population[best_idx] + mutation_factor * (a - b), self.lower_bound, self.upper_bound)
            trial = [mutant[j] if np.random.rand() < crossover_rate else population[i][j] for j in range(self.dim)]
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
        return new_population, new_fitness