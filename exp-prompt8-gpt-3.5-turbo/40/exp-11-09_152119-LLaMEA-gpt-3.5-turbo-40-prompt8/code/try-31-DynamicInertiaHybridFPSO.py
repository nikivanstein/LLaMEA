import numpy as np

class DynamicInertiaHybridFPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iter = budget // self.population_size
        self.explore_prob = 0.5  # Initial exploration probability

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        
        def dynamic_inertia(inertia_weight, global_best_pos, curr_pos, best_pos):
            inertia_weight = 1 / (1 + np.linalg.norm(global_best_pos - curr_pos))
            velocity = inertia_weight * np.zeros(self.dim)
            cognitive_weight = 1.5
            social_weight = 1.5
            velocity += cognitive_weight * np.random.rand() * (best_pos - curr_pos) + social_weight * np.random.rand() * (global_best_pos - curr_pos)
            return curr_pos + velocity, inertia_weight
        
        population = initialize_population()
        global_best_pos = population[np.argmin([func(ind) for ind in population])]
        inertia_weight = 0.7
        
        for _ in range(self.max_iter):
            for i in range(self.population_size):
                if np.random.rand() < self.explore_prob:
                    population[i], inertia_weight = dynamic_inertia(inertia_weight, global_best_pos, population[i], global_best_pos)
                else:
                    velocity = np.zeros(self.dim)
                    velocity += inertia_weight * np.random.rand() * (global_best_pos - population[i])
                    population[i] = population[i] + velocity
                
                if func(population[i]) < func(global_best_pos):
                    global_best_pos = population[i]
            
            self.explore_prob = 0.5 * (1 - _ / self.max_iter)  # Adapt exploration probability
            
        return global_best_pos