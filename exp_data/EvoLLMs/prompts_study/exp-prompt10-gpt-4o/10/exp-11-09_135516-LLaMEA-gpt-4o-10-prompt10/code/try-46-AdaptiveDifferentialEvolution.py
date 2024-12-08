import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4 * self.dim, 20)  # Adjusted population size for better balance
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.elite_archive = []  # Archive for storing elite solutions

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

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Dynamic adaptation of mutation factor and crossover rate
                F = 0.4 + np.random.rand() * 0.6  # Slightly wider range for mutation factor
                CR = 0.7 + np.random.rand() * 0.3  # Slightly wider range for crossover rate

                # Mutation with elite archive consideration
                if self.elite_archive and np.random.rand() < 0.1:
                    random_elite = self.elite_archive[np.random.randint(len(self.elite_archive))]
                    indices = np.random.choice([x for x in range(self.population_size) if x != i], 2, replace=False)
                    a, b = self.population[indices]
                    mutant = np.clip(random_elite + F * (a - b), self.lower_bound, self.upper_bound)
                else:
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
                        if trial_fitness < best_fitness:
                            self.elite_archive.append(trial)  # Update the elite archive
                if self.evaluations >= self.budget:
                    break

            # Restart strategy to escape local optima
            if self.evaluations < self.budget and self.evaluations % (self.budget // 10) == 0:
                self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                self.fitness = np.full(self.population_size, np.inf)

        return best_solution