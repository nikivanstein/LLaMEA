import numpy as np

class EnhancedHybridFPSO_Elitism:
    def __init__(self, budget, dim, elite_frac=0.2):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iter = budget // self.population_size
        self.explore_prob = 0.5  # Initial exploration probability
        self.mutation_rate = 0.5  # Initial mutation rate
        self.elite_frac = elite_frac

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        
        def dynamic_mutation(individual, best_pos, global_best_pos):
            mutation_strength = self.mutation_rate / (1 + np.linalg.norm(individual - global_best_pos))
            return individual + mutation_strength * np.random.normal(0, 1, size=self.dim)
        
        def swarm_move(curr_pos, best_pos, global_best_pos):
            inertia_weight = 0.7
            cognitive_weight = 1.5
            social_weight = 1.5
            velocity = np.zeros(self.dim)
            velocity = inertia_weight * velocity + cognitive_weight * np.random.rand() * (best_pos - curr_pos) + social_weight * np.random.rand() * (global_best_pos - curr_pos)
            return curr_pos + velocity
        
        population = initialize_population()
        global_best_pos = population[np.argmin([func(ind) for ind in population])]
        
        for _ in range(self.max_iter):
            new_population = []
            elites_count = int(self.elite_frac * self.population_size)
            elites = population[np.argsort([func(ind) for ind in population])[:elites_count]]
            
            for i in range(self.population_size):
                if i < elites_count:
                    new_population.append(elites[i])
                    continue
                
                if np.random.rand() < self.explore_prob:
                    new_individual = dynamic_mutation(population[i], global_best_pos, global_best_pos)
                else:
                    new_individual = swarm_move(population[i], population[i], global_best_pos)
                
                if func(new_individual) < func(global_best_pos):
                    global_best_pos = new_individual
                new_population.append(new_individual)
            
            population = np.array(new_population)
            self.mutation_rate *= 0.95  # Update mutation rate
            self.explore_prob = 0.5 * (1 - _ / self.max_iter)  # Adapt exploration probability
            
        return global_best_pos