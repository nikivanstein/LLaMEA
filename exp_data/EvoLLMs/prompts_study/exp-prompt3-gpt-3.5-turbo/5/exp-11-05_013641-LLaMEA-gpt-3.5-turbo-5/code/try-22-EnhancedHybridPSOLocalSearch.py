import numpy as np

class EnhancedHybridPSOLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_velocity = 0.2 * (5.0 - (-5.0))
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.local_search_radius = 0.1 * (5.0 - (-5.0))

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(low=-5.0, high=5.0, size=(self.population_size, self.dim))

        def local_search(current_position):
            candidate_position = current_position + np.random.uniform(low=-self.local_search_radius, high=self.local_search_radius, size=self.dim)
            return candidate_position

        def optimize():
            population = initialize_population()
            personal_best = population.copy()
            global_best_idx = np.argmin([func(ind) for ind in population])
            global_best = population[global_best_idx].copy()

            for _ in range(self.budget):
                velocities = np.random.uniform(low=-self.max_velocity, high=self.max_velocity, size=(self.population_size, self.dim))
                for i in range(self.population_size):
                    cognitive_weight = 1.5 * (1 - _ / self.budget)  # Dynamic adjustment of cognitive weight
                    social_weight = 1.5 * (_ / self.budget)  # Dynamic adjustment of social weight
                    velocities[i] = self.inertia_weight * velocities[i] + cognitive_weight * np.random.rand() * (personal_best[i] - population[i]) + social_weight * np.random.rand() * (global_best - population[i])
                    population[i] += velocities[i]
                    population[i] = np.clip(population[i], -5.0, 5.0)

                    if func(population[i]) < func(personal_best[i]):
                        personal_best[i] = population[i].copy()
                        if func(personal_best[i]) < func(global_best):
                            global_best = personal_best[i].copy()

                    population[i] = local_search(population[i])

            return global_best

        return optimize()