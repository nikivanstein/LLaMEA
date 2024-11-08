import numpy as np

class PSO_SA_Optimizer_Enhanced_V6:
    def __init__(self, budget, dim, pop_size=30):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.85  # Adjusted max inertia weight for quicker adaptation
        self.w_min = 0.35  # Adjusted min inertia weight for quicker adaptation
        self.c1_init = 1.4  # Tuning cognitive coefficient for exploration
        self.c2_init = 1.6  # Tuning social coefficient for exploitation
        self.temp_init = 1.3  # Higher initial temperature for exploration

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-0.4, 0.4, (self.pop_size, self.dim))  # Adjusted velocity range
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        eval_count = self.pop_size
        temperature = self.temp_init
        
        while eval_count < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * (eval_count / self.budget))
            c1 = self.c1_init * (1 - eval_count / self.budget) + 0.55
            c2 = self.c2_init * (eval_count / self.budget) + 0.45
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (personal_best_positions[i] - positions[i])
                                 + c2 * r2 * (global_best_position - positions[i]))
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

                score = func(positions[i])
                eval_count += 1
                if eval_count >= self.budget:
                    break

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                if np.random.rand() < np.exp(-(score - global_best_score) / temperature):
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = positions[i]

            temperature *= 0.88  # Fine-tuned annealing factor for adaptive temperature decay

        return global_best_position, global_best_score