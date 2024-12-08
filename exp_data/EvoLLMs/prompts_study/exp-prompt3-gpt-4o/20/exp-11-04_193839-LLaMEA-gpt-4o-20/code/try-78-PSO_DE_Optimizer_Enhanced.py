import numpy as np

class PSO_DE_Optimizer_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.inertia_weight = 0.5  # Adjusted from 0.6
        self.cognitive_coeff = 1.5  # Adjusted from 1.7
        self.social_coeff = 1.6  # Adjusted from 1.4
        self.mutation_factor = 0.6  # Adjusted for better exploration
        self.crossover_prob_mean = 0.8  # Adjusted from 0.7
        self.crossover_prob = 0.9
        self.mu_mutation_factor = 0.5
        self.mu_crossover_prob = 0.5

    def __call__(self, func):
        num_evaluations = 0

        # Initialize population for PSO
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')
        
        # Initialize population for DE
        de_population = np.copy(positions)
        fitness_diversity_threshold = 0.2  # Adjusted from 0.1
        
        while num_evaluations < self.budget:
            # PSO Part
            for i in range(self.population_size):
                score = func(positions[i])
                num_evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            # Update velocities and positions with adaptive inertia
            inertia_weight = 0.9 - 0.5 * (num_evaluations / self.budget)  # Adaptive inertia weight
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - positions) +
                          self.social_coeff * r2 * (global_best_position - positions))
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)
            
            # JADE-inspired DE Part with diversity boost
            unique_scores = np.unique(personal_best_scores)
            if len(unique_scores) > fitness_diversity_threshold * self.population_size:
                for i in range(self.population_size):
                    idx = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                    current_mutation_factor = np.clip(np.random.normal(self.mu_mutation_factor, 0.05), 0, 1)
                    mutant_vector = de_population[idx[0]] + current_mutation_factor * (de_population[idx[1]] - de_population[idx[2]])
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                    
                    current_crossover_prob = np.clip(np.random.normal(self.mu_crossover_prob, 0.05), 0, 1)
                    trial_vector = np.copy(de_population[i])
                    crossover = np.random.rand(self.dim) < current_crossover_prob
                    trial_vector[crossover] = mutant_vector[crossover]
                    
                    trial_score = func(trial_vector)
                    num_evaluations += 1
                    if trial_score < func(de_population[i]):
                        de_population[i] = trial_vector
                        self.mu_mutation_factor = (1 - 0.1) * self.mu_mutation_factor + 0.1 * current_mutation_factor
                        self.mu_crossover_prob = (1 - 0.1) * self.mu_crossover_prob + 0.1 * current_crossover_prob

                    if num_evaluations >= self.budget:
                        break

        return global_best_position, global_best_score