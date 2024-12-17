import numpy as np

class MultiStrategyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.w = 0.7  # inertia weight
        self.c1 = 1.4  # cognitive parameter
        self.c2 = 1.4  # social parameter
        self.bounds = (-5.0, 5.0)
        self.mutation_prob = 0.1
        self.mutation_scale = 0.5
        
    def __call__(self, func):
        lower, upper = self.bounds
        pop = np.random.uniform(lower, upper, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.array([func(ind) for ind in pop])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        evaluations = self.population_size

        while evaluations < self.budget:
            # Update velocities and positions
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.w * velocities + 
                          self.c1 * r1 * (personal_best_positions - pop) + 
                          self.c2 * r2 * (global_best_position - pop))
            pop = pop + velocities
            # Apply bounds
            pop = np.clip(pop, lower, upper)

            # Evaluate population
            scores = np.array([func(ind) for ind in pop])
            evaluations += self.population_size

            # Update personal bests
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best_positions[better_mask] = pop[better_mask]

            # Update global best
            min_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_idx] < global_best_score:
                global_best_score = personal_best_scores[min_idx]
                global_best_position = personal_best_positions[min_idx]
            
            # Apply adaptive mutation
            if np.random.rand() < self.mutation_prob:
                mutation_strength = np.abs(global_best_score - scores) / (np.max(scores) - np.min(scores) + 1e-12)
                for i in range(self.population_size):
                    if np.random.rand() < mutation_strength[i]:
                        mutation_vector = np.random.normal(0, self.mutation_scale, self.dim)
                        pop[i] = np.clip(pop[i] + mutation_vector, lower, upper)
                        new_score = func(pop[i])
                        evaluations += 1
                        if new_score < personal_best_scores[i]:
                            personal_best_scores[i] = new_score
                            personal_best_positions[i] = pop[i]
                            if new_score < global_best_score:
                                global_best_score = new_score
                                global_best_position = pop[i]

        return global_best_position, global_best_score