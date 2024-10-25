import numpy as np

class AdaptiveHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(6 * dim, 60)  # slightly increased adaptive population size
        self.F = 0.5  # differential weight adjusted for balance
        self.CR = 0.8  # crossover probability slightly reduced for exploration
        self.w = 0.5  # inertia weight for PSO influence
        self.c1 = 1.5  # cognitive (self) component
        self.c2 = 1.5  # social component

    def __call__(self, func):
        # Initialize population and velocities
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size
        pbest = population.copy()
        pbest_fitness = fitness.copy()
        gbest_idx = np.argmin(fitness)
        gbest = population[gbest_idx].copy()

        while evals < self.budget:
            # Differential Evolution Phase with PSO influence
            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                F_dynamic = self.F if np.random.rand() < 0.6 else np.random.uniform(0.4, 0.9)
                mutant = np.clip(a + F_dynamic * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # PSO Update
                velocities[i] = self.w * velocities[i] + \
                                self.c1 * np.random.rand(self.dim) * (pbest[i] - population[i]) + \
                                self.c2 * np.random.rand(self.dim) * (gbest - population[i])
                trial = np.clip(trial + velocities[i], self.lower_bound, self.upper_bound)

                # Selection
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    pbest[i] = trial
                    pbest_fitness[i] = trial_fitness
                    if trial_fitness < fitness[gbest_idx]:
                        gbest_idx = i
                        gbest = trial

            # Dynamic Local Search Phase
            for _ in range(4 + int(0.1 * self.dim)):
                if evals >= self.budget:
                    break

                potential_neighbor = gbest + np.random.normal(0, 0.05, self.dim)
                potential_neighbor = np.clip(potential_neighbor, self.lower_bound, self.upper_bound)
                neighbor_fitness = func(potential_neighbor)
                evals += 1

                if neighbor_fitness < pbest_fitness[gbest_idx]:
                    gbest = potential_neighbor
                    pbest_fitness[gbest_idx] = neighbor_fitness

            # Adaptive Diversity Enhancement
            if evals < 0.6 * self.budget and len(population) < 2 * self.population_size:
                new_individuals = np.random.uniform(self.lower_bound, self.upper_bound, (3, self.dim))
                new_fitness = np.array([func(ind) for ind in new_individuals])
                evals += len(new_individuals)
                population = np.vstack((population, new_individuals))
                fitness = np.hstack((fitness, new_fitness))

        return gbest