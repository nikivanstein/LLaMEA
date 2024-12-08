import numpy as np

class EnhancedParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 * dim
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_min = 0.4
        self.w_max = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight = self.w_max

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def _get_fitness(self, population, func):
        return np.array([func(individual) for individual in population])

    def _update_velocity(self, velocity, position, personal_best, global_best):
        r1 = np.random.uniform(0, 1, self.dim)
        r2 = np.random.uniform(0, 1, self.dim)
        return self.inertia_weight * velocity + self.c1 * r1 * (personal_best - position) + self.c2 * r2 * (global_best - position)

    def _update_position(self, position, velocity):
        new_position = position + velocity
        return np.clip(new_position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        population = self._initialize_population()
        velocities = np.zeros((self.pop_size, self.dim))
        personal_best = population.copy()
        global_best = population[np.argmin(self._get_fitness(population, func))]
        evals = 0

        while evals < self.budget:
            for i in range(self.pop_size):
                velocities[i] = self._update_velocity(velocities[i], population[i], personal_best[i], global_best)
                new_position = self._update_position(population[i], velocities[i])
                population[i] = new_position
                evals += 1

                if evals >= self.budget:
                    break

            fitness_values = self._get_fitness(population, func)
            personal_best = np.where(fitness_values < self._get_fitness(personal_best, func), population, personal_best)
            global_best = population[np.argmin(fitness_values)]

            # Dynamic inertia weight
            self.inertia_weight = self.w_max - (self.w_max - self.w_min) * evals / self.budget

        best_solution = global_best
        return best_solution