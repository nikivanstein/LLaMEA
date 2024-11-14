import numpy as np

class EnhancedHybridFPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iter = budget // self.population_size
        self.explore_prob = 0.5  # Initial exploration probability
        self.mutation_rate = 0.5  # Initial mutation rate
        self.inertia_range = [0.4, 0.9]  # Range for inertia weight
        self.diversity_rate = 0.1  # Rate for diversity calculation
        
    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        
        def update_inertia(population):
            diversity = np.std(population, axis=0)
            inertia_weight = self.inertia_range[0] + (self.inertia_range[1] - self.inertia_range[0]) * (1 - np.mean(diversity) / np.max(diversity))
            return inertia_weight
        
        def dynamic_mutation(individual, best_pos, global_best_pos):
            mutation_strength = self.mutation_rate / (1 + np.linalg.norm(individual - global_best_pos))
            return individual + mutation_strength * np.random.normal(0, 1, size=self.dim)
        
        def swarm_move(curr_pos, best_pos, global_best_pos, inertia_weight):
            velocity = np.zeros(self.dim)
            velocity = inertia_weight * velocity + np.random.rand() * (best_pos - curr_pos) + np.random.rand() * (global_best_pos - curr_pos)
            return curr_pos + velocity
        
        population = initialize_population()
        global_best_pos = population[np.argmin([func(ind) for ind in population])]
        
        for _ in range(self.max_iter):
            inertia_weight = update_inertia(population)
            for i in range(self.population_size):
                if np.random.rand() < self.explore_prob:
                    population[i] = dynamic_mutation(population[i], global_best_pos, global_best_pos)
                else:
                    population[i] = swarm_move(population[i], population[i], global_best_pos, inertia_weight)
                
                if func(population[i]) < func(global_best_pos):
                    global_best_pos = population[i]
            
            self.mutation_rate *= 0.95  # Update mutation rate
            self.explore_prob = 0.5 * (1 - _ / self.max_iter)  # Adapt exploration probability
            
        return global_best_pos