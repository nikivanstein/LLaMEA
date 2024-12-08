import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = max(4 * self.dim, 20)
        self.population_size = self.initial_population_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf

        # Initial fitness evaluation
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.fitness[i] < best_fitness:
                best_fitness = self.fitness[i]
                best_solution = self.population[i]

        def local_search(solution, fitness):
            step_size = 0.1  # Small step size for local exploration
            for _ in range(5):  # Perform a few local exploration steps
                candidate = np.clip(solution + np.random.uniform(-step_size, step_size, self.dim), self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                self.evaluations += 1
                if candidate_fitness < fitness:
                    solution, fitness = candidate, candidate_fitness
                    if candidate_fitness < best_fitness:
                        return candidate, candidate_fitness
                if self.evaluations >= self.budget:
                    break
            return solution, fitness

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Dynamic adaptation of mutation factor and crossover rate
                F = 0.5 + np.random.rand() * 0.5  # Slightly narrowed range for mutation factor
                CR = 0.8 + np.random.rand() * 0.2  # Slightly narrowed range for crossover rate

                # Mutation
                indices = np.random.choice([x for x in range(self.population_size) if x != i], 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < CR
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(self.dim)] = True  # Ensure at least one crossover point
                trial[crossover_points] = mutant[crossover_points]

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial

                # Local search on the trial solution
                if self.evaluations < self.budget:
                    self.population[i], self.fitness[i] = local_search(self.population[i], self.fitness[i])

                if self.evaluations >= self.budget:
                    break

            # Dynamic adjustment of population size
            if self.evaluations / self.budget > 0.5:  # Reduce population size after half the budget is used
                self.population_size = max(self.initial_population_size // 2, 5)
                self.population = self.population[:self.population_size]
                self.fitness = self.fitness[:self.population_size]

        return best_solution