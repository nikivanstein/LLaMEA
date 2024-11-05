class AdaptiveInertiaHybridPSOLocalSearch(HybridPSOLocalSearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.initial_inertia_weight = 0.7

    def __call__(self, func):
        def optimize():
            inertia_weight = self.initial_inertia_weight
            population = initialize_population()
            personal_best = population.copy()
            global_best_idx = np.argmin([func(ind) for ind in population])
            global_best = population[global_best_idx].copy()

            for _ in range(self.budget):
                velocities = np.random.uniform(low=-self.max_velocity, high=self.max_velocity, size=(self.population_size, self.dim))
                for i in range(self.population_size):
                    velocities[i] = inertia_weight * velocities[i] + self.cognitive_weight * np.random.rand() * (personal_best[i] - population[i]) + self.social_weight * np.random.rand() * (global_best - population[i])
                    population[i] += velocities[i]
                    population[i] = np.clip(population[i], -5.0, 5.0)

                    if func(population[i]) < func(personal_best[i]):
                        personal_best[i] = population[i].copy()
                        if func(personal_best[i]) < func(global_best):
                            global_best = personal_best[i].copy()

                    population[i] = local_search(population[i])

                inertia_weight = self.update_inertia_weight(inertia_weight, func(global_best))

            return global_best
        
        def update_inertia_weight(inertia_weight, best_fitness):
            if best_fitness < func(global_best):
                return max(0.4, inertia_weight - 0.05)
            else:
                return min(0.9, inertia_weight + 0.05)

        return optimize()