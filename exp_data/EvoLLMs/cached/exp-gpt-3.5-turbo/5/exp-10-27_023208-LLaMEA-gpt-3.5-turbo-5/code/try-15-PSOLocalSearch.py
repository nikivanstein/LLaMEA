import numpy as np

class PSOLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.inertia_weight = 0.5
        self cognitive_weight = 1.5
        self social_weight = 1.5
        self.local_search_prob = 0.1

    def _local_search(self, current_solution, func):
        best_solution = current_solution
        for _ in range(5):
            candidate_solution = current_solution + np.random.uniform(-0.1, 0.1, self.dim)
            if func(candidate_solution) < func(best_solution):
                best_solution = candidate_solution
        return best_solution

    def _update_velocity_position(self, population, velocities, local_best_positions, global_best_position):
        for i in range(self.population_size):
            velocities[i] = self.inertia_weight * velocities[i] + \
                             self.cognitive_weight * np.random.rand() * (local_best_positions[i] - population[i]) + \
                             self.social_weight * np.random.rand() * (global_best_position - population[i])
            population[i] = population[i] + velocities[i]
            if np.random.rand() < self.local_search_prob:
                population[i] = self._local_search(population[i])

    def _optimize_func(self, func, population):
        velocities = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
        local_best_positions = population.copy()
        global_best_position = population[0]
        for _ in range(self.budget):
            self._update_velocity_position(population, velocities, local_best_positions, global_best_position)
            for i in range(self.population_size):
                if func(population[i]) < func(local_best_positions[i]):
                    local_best_positions[i] = population[i]
                if func(population[i]) < func(global_best_position):
                    global_best_position = population[i]
        return global_best_position

    def __call__(self, func):
        population = np.random.uniform(-5, 5, (self.population_size, self.dim))
        return self._optimize_func(func, population)