import numpy as np

class HybridFireflySAOptimizer:
    def __init__(self, budget, dim, population_size=20, alpha=0.9, initial_temp=10.0, final_temp=0.1, mutation_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.alpha = alpha
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def move_fireflies(fireflies, best_firefly):
            new_fireflies = []
            for firefly in fireflies:
                attractiveness = 1 / (1 + np.linalg.norm(best_firefly - firefly))
                new_firefly = firefly + attractiveness * (best_firefly - firefly) + np.random.normal(0, 1, self.dim)
                new_fireflies.append(new_firefly)
            return np.array(new_fireflies)

        def simulated_annealing(current, best, temp):
            candidate = current + np.random.uniform(-1, 1, self.dim) * temp
            candidate_fitness = func(candidate)
            current_fitness = func(current)
            if candidate_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - candidate_fitness) / temp):
                return candidate
            return current

        def mutate_firefly(firefly):
            mutated_firefly = firefly + np.random.normal(0, 1, self.dim)
            return mutated_firefly

        fireflies = initialize_population()
        best_firefly = fireflies[np.argmin([func(f) for f in fireflies])]
        temperature = self.initial_temp
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0 and temperature > self.final_temp:
            new_fireflies = move_fireflies(fireflies, best_firefly)
            for idx, firefly in enumerate(new_fireflies):
                new_firefly = simulated_annealing(firefly, best_firefly, temperature)
                new_firefly = mutate_firefly(new_firefly)
                new_fitness = func(new_firefly)
                if new_fitness < func(fireflies[idx]):
                    fireflies[idx] = new_firefly
                    if new_fitness < func(best_firefly):
                        best_firefly = new_firefly
                remaining_budget -= 1
                if remaining_budget <= 0 or temperature <= self.final_temp:
                    break
            temperature *= self.alpha

        return best_firefly

# Example usage:
# optimizer = HybridFireflySAOptimizer(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function