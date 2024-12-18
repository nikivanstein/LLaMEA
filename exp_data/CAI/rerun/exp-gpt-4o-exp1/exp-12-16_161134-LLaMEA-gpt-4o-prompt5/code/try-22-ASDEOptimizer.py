import numpy as np

class ASDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0

    def __call__(self, func):
        population_size = int(min(50, self.budget // 10))
        cognitive_coeff_base = 1.5
        social_coeff_base = 1.5

        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(population_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        while self.evaluations < self.budget:
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break
                score = func(swarm[i])
                self.evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i]
                    
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm[i]

            # Update velocities and positions
            for i in range(population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                inertia_weight = 0.9 - 0.4 * (self.evaluations / self.budget)
                
                decay_factor = 0.97 + 0.01 * (self.evaluations / self.budget)
                cognitive_coeff = cognitive_coeff_base * (1 - self.evaluations / self.budget) * decay_factor
                social_coeff = social_coeff_base * (self.evaluations / self.budget) * decay_factor

                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_coeff * r1 * (personal_best_positions[i] - swarm[i]) +
                                 social_coeff * r2 * (global_best_position - swarm[i]))

                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)
                
                if np.random.rand() < 0.01:  # Random restart strategy for diversity
                    swarm[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        return {"best_position": global_best_position, "best_score": global_best_score}