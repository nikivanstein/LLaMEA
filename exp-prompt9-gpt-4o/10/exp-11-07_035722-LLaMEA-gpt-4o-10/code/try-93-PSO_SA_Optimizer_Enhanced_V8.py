import numpy as np

class PSO_SA_Optimizer_Enhanced_V8:
    def __init__(self, budget, dim, pop_size=30):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        # Adaptive inertia weight range for better exploration versus exploitation
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_init = 1.4  # Modified cognitive coefficient for enhanced local search
        self.c2_init = 1.6  # Modified social coefficient for enhanced global search
        self.temp_init = 1.25  # Slightly increased initial temperature for broader search

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-0.4, 0.4, (self.pop_size, self.dim))  # Slightly adjusted velocity range
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        eval_count = self.pop_size
        temperature = self.temp_init
        
        while eval_count < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * (eval_count / self.budget))
            c1 = self.c1_init * (1 - eval_count / self.budget) + 0.65
            c2 = self.c2_init * (eval_count / self.budget) + 0.35
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

                if i % 2 == 0:  # Hybrid mutation strategy applied every alternate particle
                    mutation = np.random.normal(0, 0.1, self.dim)
                    mutant_position = positions[i] + mutation
                    mutant_position = np.clip(mutant_position, self.lower_bound, self.upper_bound)
                    mutant_score = func(mutant_position)
                    eval_count += 1
                    if mutant_score < personal_best_scores[i]:
                        personal_best_scores[i] = mutant_score
                        personal_best_positions[i] = mutant_position
                        if mutant_score < global_best_score:
                            global_best_score = mutant_score
                            global_best_position = mutant_position

            temperature *= 0.83  # Increased cooling rate for more robust convergence

        return global_best_position, global_best_score