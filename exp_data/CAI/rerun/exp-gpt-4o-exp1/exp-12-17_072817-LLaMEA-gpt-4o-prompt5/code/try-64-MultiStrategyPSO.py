import numpy as np

class MultiStrategyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 35  # Reduced population size for faster convergence
        self.w = 0.9  # inertia weight, increased initial value
        self.c1 = 1.7  # cognitive parameter, slight tweak
        self.c2 = 1.5  # social parameter, slight tweak
        self.bounds = (-5.0, 5.0)
        self.mutation_prob = 0.1
        self.mutation_scale = 0.5
        self.local_topology = 5  # local topology size (changed from 3 to 5)
    
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
            self.w = 0.4 + (0.5 * np.random.rand()) 
            eval_ratio = evaluations / self.budget
            self.c1 = 2.5 * (1 - eval_ratio) + 0.5 * eval_ratio
            self.c2 = 0.5 * (1 - eval_ratio) + 2.5 * eval_ratio
            
            # Update velocities and positions
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.w * velocities + 
                          self.c1 * r1 * (personal_best_positions - pop) + 
                          self.c2 * r2 * (global_best_position - pop))
            velocities = np.clip(velocities, -0.7 + 0.3 * eval_ratio, 0.7 - 0.3 * eval_ratio)  # Adaptive velocity clipping
            pop = pop + velocities
            pop = np.clip(pop, lower, upper)

            # Evaluate population
            scores = np.array([func(ind) for ind in pop])
            evaluations += self.population_size

            # Update personal bests
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best_positions[better_mask] = pop[better_mask]

            # Dynamic neighborhood topology
            for i in range(self.population_size):
                local_indices = np.random.choice(self.population_size, self.local_topology, replace=False)
                local_best_idx = local_indices[np.argmin(personal_best_scores[local_indices])]
                if personal_best_scores[local_best_idx] < personal_best_scores[i]:
                    personal_best_positions[i] = personal_best_positions[local_best_idx]

            # Update global best
            min_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_idx] < global_best_score:
                global_best_score = personal_best_scores[min_idx]
                global_best_position = personal_best_positions[min_idx]

            # Apply adaptive mutation
            self.mutation_prob *= 0.95 * (1 - eval_ratio)  # Changed mutation probability decay rate
            if np.random.rand() < self.mutation_prob:
                mutation_strength = np.abs(global_best_score - scores) / (np.std(scores) + 1e-12)
                self.mutation_scale = max(0.05, self.mutation_scale * 0.98 * (1 + eval_ratio))  # Adaptive mutation scale
                diversity = np.std(pop, axis=0).mean()  # Measure of swarm diversity
                for i in range(self.population_size):
                    if np.random.rand() < mutation_strength[i] + diversity * 0.1:  # Enhance mutation chance
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