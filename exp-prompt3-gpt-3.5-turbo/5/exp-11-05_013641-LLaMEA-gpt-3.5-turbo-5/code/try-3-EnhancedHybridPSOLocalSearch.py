import numpy as np

class EnhancedHybridPSOLocalSearch(HybridPSOLocalSearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.diversity_weight = 0.5

    def __call__(self, func):
        def optimize():
            population = initialize_population()
            personal_best = population.copy()
            global_best_idx = np.argmin([func(ind) for ind in population])
            global_best = population[global_best_idx].copy()
            
            inertia_weights = np.full(self.population_size, self.inertia_weight)

            for _ in range(self.budget):
                velocities = np.random.uniform(low=-self.max_velocity, high=self.max_velocity, size=(self.population_size, self.dim))
                avg_distance = np.mean([np.linalg.norm(population - np.mean(population, axis=0)) for population in population])
                diversity = np.exp(-avg_distance)

                for i in range(self.population_size):
                    inertia_weights[i] = self.inertia_weight + self.diversity_weight * diversity
                    velocities[i] = inertia_weights[i] * velocities[i] + self.cognitive_weight * np.random.rand() * (personal_best[i] - population[i]) + self.social_weight * np.random.rand() * (global_best - population[i])
                    population[i] += velocities[i]
                    population[i] = np.clip(population[i], -5.0, 5.0)

                    if func(population[i]) < func(personal_best[i]):
                        personal_best[i] = population[i].copy()
                        if func(personal_best[i]) < func(global_best):
                            global_best = personal_best[i].copy()

                    population[i] = local_search(population[i])

            return global_best

        return optimize()