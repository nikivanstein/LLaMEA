import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.min_population_size = 5
        self.max_population_size = 20

    def __call__(self, func):
        population_size = self.min_population_size
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        velocities = np.zeros((population_size, self.dim))
        inertia_weight = 0.8
        mutation_rate = 0.2

        for _ in range(self.budget):
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

            for i in range(population_size):
                velocity = velocities[i]
                inertia_weight *= 0.99
                cognitive_component = 1.5 * np.random.random() * (harmony_memory[i] - new_solution)
                social_component = 1.5 * np.random.random() * (harmony_memory[np.random.randint(population_size)] - new_solution)
                
                # Introducing differential evolution mutation
                mutant = harmony_memory[np.random.choice(np.arange(population_size), replace=False, size=3)]
                differential_component = mutation_rate * np.sum(mutant, axis=0) / 3
                mutation_component = np.where(np.random.rand(self.dim) < mutation_rate, differential_component, 0.0)
                
                velocity = inertia_weight * velocity + cognitive_component + social_component + mutation_component
                new_solution = np.clip(new_solution + velocity, self.lower_bound, self.upper_bound)

                if func(new_solution) < func(harmony_memory[i]):
                    harmony_memory[i] = new_solution

            if (population_size < self.max_population_size) and np.random.rand() < 0.1:
                population_size += 1
                harmony_memory = np.vstack((harmony_memory, np.random.uniform(self.lower_bound, self.upper_bound, self.dim)))
                velocities = np.vstack((velocities, np.zeros(self.dim)))

        best_solution = min(harmony_memory, key=func)
        return best_solution