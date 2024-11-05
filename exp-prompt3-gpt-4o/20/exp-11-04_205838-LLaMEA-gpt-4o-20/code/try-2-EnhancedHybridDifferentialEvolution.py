import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.scaling_factor = 0.8
        self.crossover_rate = 0.9
        self.inertia_weight = 0.7  # Added for swarm behavior
        self.cognitive_component = 1.5  # Personal best influence
        self.social_component = 1.5  # Global best influence

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))  # Initialize velocities
        
        cov_matrix = np.eye(self.dim)

        while num_evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] + 
                                 self.cognitive_component * r1 * (personal_best[i] - population[i]) +
                                 self.social_component * r2 * (best_individual - population[i]))
                velocities[i] = np.clip(velocities[i], self.lower_bound, self.upper_bound)
                
                trial_vector = population[i] + velocities[i]
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                if func(trial_vector) < personal_best_fitness[i]:
                    personal_best[i] = trial_vector
                    personal_best_fitness[i] = func(trial_vector)
            
            trial_fitness = np.array([func(ind) for ind in personal_best])
            num_evaluations += self.population_size
            
            improvement_mask = trial_fitness < fitness
            population[improvement_mask] = personal_best[improvement_mask]
            fitness[improvement_mask] = trial_fitness[improvement_mask]

            new_best_idx = np.argmin(fitness)
            if fitness[new_best_idx] < func(best_individual):
                best_individual = population[new_best_idx]

            if num_evaluations < self.budget:
                sampled_indices = np.random.choice(self.population_size, 2, replace=False)
                diff = population[sampled_indices[0]] - population[sampled_indices[1]]
                cov_matrix = (1 - 1.0/self.dim) * cov_matrix + (1.0/self.dim) * np.outer(diff, diff)

                self.scaling_factor = self.adjust_scaling_factor(cov_matrix)

        return best_individual

    def adjust_scaling_factor(self, cov_matrix):
        variance_measure = np.sum(np.diag(cov_matrix))
        return min(1.0, max(0.5, 0.8 * (1.0 + variance_measure)))