import numpy as np

class APSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(max(20, dim * 5), 100)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.4
        self.vel_clamp = (-(self.upper_bound - self.lower_bound), (self.upper_bound - self.lower_bound))
        self.eval_count = 0

    def __call__(self, func):
        np.random.seed(42)  # for reproducibility
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(self.vel_clamp[0], self.vel_clamp[1], (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break

                score = func(positions[i])
                self.eval_count += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                cognitive_velocity = self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.social_coeff * r2 * (global_best_position - positions[i])
                velocities[i] = (self.inertia_weight * velocities[i]) + cognitive_velocity + social_velocity
                
                # Clamp velocity
                velocities[i] = np.clip(velocities[i], self.vel_clamp[0], self.vel_clamp[1])

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

        return global_best_position, global_best_score