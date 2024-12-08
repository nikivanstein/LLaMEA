import numpy as np

class DynamicMutationVelocityGuidedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 10
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        velocities = np.zeros((population_size, self.dim))
        inertia_weight = 0.8
        mutation_factor = 1.0

        for eval_count in range(1, self.budget + 1):
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

            for i in range(population_size):
                velocity = velocities[i]
                inertia_weight *= 0.99
                cognitive_component = 1.5 * np.random.random() * (harmony_memory[i] - new_solution)
                social_component = 1.5 * np.random.random() * (harmony_memory[np.random.randint(population_size)] - new_solution)
                velocity = inertia_weight * velocity + cognitive_component + social_component
                mutation_step = np.exp(-eval_count / self.budget)  # Dynamic mutation step
                new_solution = np.clip(new_solution + mutation_factor * mutation_step * velocity, self.lower_bound, self.upper_bound)

                if func(new_solution) < func(harmony_memory[i]):
                    harmony_memory[i] = new_solution

        best_solution = min(harmony_memory, key=func)
        return best_solution