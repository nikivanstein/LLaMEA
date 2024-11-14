import numpy as np

class EnhancedHarmonyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_swarm_size = 50 + (dim * 5)
        self.position = np.random.uniform(-5.0, 5.0, (self.initial_swarm_size, dim))
        self.velocity = np.zeros((self.initial_swarm_size, dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.initial_swarm_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.w_max = 0.8
        self.w_min = 0.3
        self.c1 = 2.5
        self.c2 = 2.5
        self.harmony_memory_consideration_rate = 0.9
        self.pitch_adjustment_rate = 0.1
        self.scale_factor = 0.05
        self.population_shrinkage_factor = 0.9

    def __call__(self, func):
        evals = 0
        swarm_size = self.initial_swarm_size

        while evals < self.budget:
            fitness_values = np.array([func(pos) for pos in self.position])
            evals += swarm_size

            better_mask = fitness_values < self.personal_best_value
            self.personal_best_value[better_mask] = fitness_values[better_mask]
            self.personal_best_position[better_mask] = self.position[better_mask]

            min_fitness_idx = np.argmin(fitness_values)
            if fitness_values[min_fitness_idx] < self.global_best_value:
                self.global_best_value = fitness_values[min_fitness_idx]
                self.global_best_position = self.position[min_fitness_idx]

            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (evals / self.budget))
            r1 = np.random.rand(swarm_size, self.dim)
            r2 = np.random.rand(swarm_size, self.dim)

            cognitive_component = self.c1 * r1 * (self.personal_best_position - self.position)
            social_component = self.c2 * r2 * (self.global_best_position - self.position)
            self.velocity = inertia_weight * self.velocity + cognitive_component + social_component

            harmony_decision = np.random.rand(swarm_size, self.dim) < self.harmony_memory_consideration_rate
            pitch_adjustment = (np.random.rand(swarm_size, self.dim) - 0.5) * self.scale_factor
            harmony_component = harmony_decision * pitch_adjustment
            new_position = self.position + self.velocity + harmony_component

            new_position = np.clip(new_position, -5.0, 5.0)
            self.position = new_position

            swarm_size = max(5, int(swarm_size * self.population_shrinkage_factor))

        return self.global_best_position, self.global_best_value