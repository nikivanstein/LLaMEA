import numpy as np

class EnhancedHybridPSODE(HybridPSODE):
    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        population = initialize_population()
        velocities = np.zeros((self.population_size, self.dim))
        p_best = population.copy()
        fitness = np.array([func(individual) for individual in population])
        p_best_fitness = fitness.copy()
        g_best_idx = np.argmin(fitness)
        g_best = population[g_best_idx]

        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                r1, r2 = np.random.uniform(0, 1, 2)
                velocities[i] = self.inertia_weight * velocities[i] + self.cognitive_weight * r1 * (p_best[i] - population[i]) + self.social_weight * r2 * (g_best - population[i])
                population[i] = np.clip(population[i] + velocities[i], -5.0, 5.0)

                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.scaling_factor * (b - c), -5.0, 5.0)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_mask, mutant, population[i])
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < p_best_fitness[i]:
                        p_best[i] = trial
                        p_best_fitness[i] = trial_fitness

                        if trial_fitness < fitness[g_best_idx]:
                            g_best_idx = i
                            g_best = trial

                # Introduce local search for further exploitation
                for _ in range(3):
                    local_search_candidate = np.clip(population[i] + np.random.normal(0, 0.1, self.dim), -5.0, 5.0)
                    local_search_fitness = func(local_search_candidate)
                    if local_search_fitness < fitness[i]:
                        population[i] = local_search_candidate
                        fitness[i] = local_search_fitness
                        if local_search_fitness < p_best_fitness[i]:
                            p_best[i] = local_search_candidate
                            p_best_fitness[i] = local_search_fitness
                            if local_search_fitness < fitness[g_best_idx]:
                                g_best_idx = i
                                g_best = local_search_candidate

        return g_best
