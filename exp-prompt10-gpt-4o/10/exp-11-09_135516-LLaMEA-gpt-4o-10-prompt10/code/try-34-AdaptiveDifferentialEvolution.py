import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = max(4 * self.dim, 20)  # Adjusted initial population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim))
        self.fitness = np.full(self.initial_population_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf

        # Initial fitness evaluation
        for i in range(self.initial_population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.fitness[i] < best_fitness:
                best_fitness = self.fitness[i]
                best_solution = self.population[i]

        while self.evaluations < self.budget:
            new_population = []
            new_fitness = []

            for i in range(self.initial_population_size):
                # Dynamic adaptation of mutation factor and reduced crossover rate range for tighter control
                F = 0.5 + np.random.rand() * 0.5  # Focused range for mutation factor
                CR = 0.6 + np.random.rand() * 0.2  # Reduced range for crossover rate

                # Mutation with additional diversity enhancement
                indices = np.random.choice([x for x in range(self.initial_population_size) if x != i], 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c) + 0.001 * np.random.randn(self.dim), self.lower_bound, self.upper_bound)

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
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial
                else:
                    new_population.append(self.population[i])
                    new_fitness.append(self.fitness[i])

                if self.evaluations >= self.budget:
                    break

            # Update population and fitness
            self.population = np.array(new_population)
            self.fitness = np.array(new_fitness)

        return best_solution