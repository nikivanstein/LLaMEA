import numpy as np

class EnhancedHybridFPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iter = budget // self.population_size
        self.explore_prob = 0.5  # Initial exploration probability
        self.local_search_prob = 0.2  # Probability of applying local search
        
    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        
        def dynamic_mutation(individual, best_pos):
            mutation_strength = 0.5 / np.sqrt(1 + np.linalg.norm(individual - best_pos))
            return individual + mutation_strength * np.random.normal(0, 1, size=self.dim)
        
        def local_search(individual, best_pos):
            return 0.5 * individual + 0.5 * best_pos
        
        def swarm_move(curr_pos, best_pos, global_best_pos):
            inertia_weight = 0.7
            cognitive_weight = 1.5
            social_weight = 1.5
            velocity = np.zeros(self.dim)
            velocity = inertia_weight * velocity + cognitive_weight * np.random.rand() * (best_pos - curr_pos) + social_weight * np.random.rand() * (global_best_pos - curr_pos)
            return curr_pos + velocity
        
        population = initialize_population()
        global_best_pos = population[np.argmin([func(ind) for ind in population])
        
        for _ in range(self.max_iter):
            for i in range(self.population_size):
                if np.random.rand() < self.explore_prob:
                    population[i] = dynamic_mutation(population[i], global_best_pos)
                else:
                    if np.random.rand() < self.local_search_prob:
                        population[i] = local_search(population[i], global_best_pos)
                    else:
                        population[i] = swarm_move(population[i], population[i], global_best_pos)
                
                if func(population[i]) < func(global_best_pos):
                    global_best_pos = population[i]
            
            self.explore_prob = 0.5 * (1 - _ / self.max_iter)  # Adapt exploration probability
            
        return global_best_pos