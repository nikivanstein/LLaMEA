import numpy as np

class HybridDE_VNS_PSOLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_velocity = 0.2 * (5.0 - (-5.0))
        self.initial_inertia_weight = 0.7
        self.initial_cognitive_weight = 1.5
        self.initial_social_weight = 1.5
        self.local_search_radius = 0.1 * (5.0 - (-5.0))

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(low=-5.0, high=5.0, size=(self.population_size, self.dim))

        def local_search(current_position):
            candidate_position = current_position + np.random.uniform(low=-self.local_search_radius, high=self.local_search_radius, size=self.dim)
            return candidate_position

        def variable_neighborhood_search(current_position, radius):
            candidate_position = current_position + np.random.uniform(low=-radius, high=radius, size=self.dim)
            return candidate_position

        def differential_evolution(population, func):
            mutation_factor = 0.7
            crossover_rate = 0.9
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), -5.0, 5.0)
                trial = np.where(np.random.uniform(0, 1, self.dim) < crossover_rate, mutant, population[i])
                if func(trial) < func(population[i]):
                    population[i] = trial
            return population

        def optimize():
            population = initialize_population()
            personal_best = population.copy()
            global_best_idx = np.argmin([func(ind) for ind in population])
            global_best = population[global_best_idx].copy()

            for _ in range(self.budget):
                inertia_weight = self.initial_inertia_weight * (1 - _ / self.budget)  # Dynamic inertia weight
                social_weight = self.initial_social_weight / (1 + 0.1 * np.sqrt(_))  # Adaptive social weight
                velocities = np.random.uniform(low=-self.max_velocity, high=self.max_velocity, size=(self.population_size, self.dim))
                for i in range(self.population_size):
                    velocities[i] = inertia_weight * velocities[i] + self.initial_cognitive_weight * np.random.rand() * (personal_best[i] - population[i]) + social_weight * np.random.rand() * (global_best - population[i])
                    population[i] += velocities[i]
                    population[i] = np.clip(population[i], -5.0, 5.0)

                    if func(population[i]) < func(personal_best[i]):
                        personal_best[i] = population[i].copy()
                        if func(personal_best[i]) < func(global_best):
                            global_best = personal_best[i].copy()

                    if _ % 10 == 0:  # Introduce variable neighborhood search every 10 iterations
                        population[i] = variable_neighborhood_search(population[i], self.local_search_radius)

                    if _ % 15 == 0:  # Incorporate Differential Evolution every 15 iterations for global exploration
                        population = differential_evolution(population, func)

            return global_best

        return optimize()