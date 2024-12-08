import numpy as np

class DynamicHybridPSOLocalSearch(HybridPSOLocalSearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.dynamic_weight_range = 0.5

    def __call__(self, func):
        def optimize():
            population = initialize_population()
            personal_best = population.copy()
            global_best_idx = np.argmin([func(ind) for ind in population])
            global_best = population[global_best_idx].copy()

            for _ in range(self.budget):
                velocities = np.random.uniform(low=-self.max_velocity, high=self.max_velocity, size=(self.population_size, self.dim))
                dynamic_cognitive_weight = np.random.uniform(max(0.5, self.cognitive_weight - self.dynamic_weight_range), self.cognitive_weight + self.dynamic_weight_range)
                dynamic_social_weight = np.random.uniform(max(0.5, self.social_weight - self.dynamic_weight_range), self.social_weight + self.dynamic_weight_range)
                for i in range(self.population_size):
                    velocities[i] = self.inertia_weight * velocities[i] + dynamic_cognitive_weight * np.random.rand() * (personal_best[i] - population[i]) + dynamic_social_weight * np.random.rand() * (global_best - population[i])
                    population[i] += velocities[i]
                    population[i] = np.clip(population[i], -5.0, 5.0)

                    if func(population[i]) < func(personal_best[i]):
                        personal_best[i] = population[i].copy()
                        if func(personal_best[i]) < func(global_best):
                            global_best = personal_best[i].copy()

                    population[i] = local_search(population[i])

            return global_best
        return optimize()