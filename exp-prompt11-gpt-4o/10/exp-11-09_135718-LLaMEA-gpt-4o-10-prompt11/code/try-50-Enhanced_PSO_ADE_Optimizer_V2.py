import numpy as np

class Enhanced_PSO_ADE_Optimizer_V2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 50
        self.mutation_factor = 0.85  # Slight increase in mutation factor for better exploration
        self.crossover_prob = 0.85  # Reduced crossover probability to maintain diversity
        self.inertia_weight = 0.6  # Increased inertia_weight for balanced exploration-exploitation
        self.cognitive_coeff = 2.0  # Further increase cognitive_coeff for aggressive personal search
        self.social_coeff = 1.2  # Slightly reduce social_coeff to retain swarm diversity
        self.dynamic_adjustment_freq = 8  # More frequent dynamic parameter adjustment
        self.local_search_threshold = 0.1  # Threshold for stochastic local search

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        eval_count = 0
        iteration = 0

        while eval_count < self.budget:
            scores = np.apply_along_axis(func, 1, positions)
            eval_count += self.swarm_size

            for i in range(self.swarm_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]

                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = positions[i]

            if iteration % self.dynamic_adjustment_freq == 0:
                self.inertia_weight *= 0.93  # Faster reduction in inertia
                self.mutation_factor = 1.0 if iteration < self.budget // 3 else 0.75  # More aggressive mutation adaptation

            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = (
                self.inertia_weight * velocities +
                self.cognitive_coeff * r1 * (personal_best_positions - positions) +
                self.social_coeff * r2 * (global_best_position - positions)
            )
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

            local_search_indices = np.where(np.random.rand(self.swarm_size) < self.local_search_threshold)[0]
            for i in range(self.swarm_size):
                if i in local_search_indices:
                    random_search = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    if func(random_search) < personal_best_scores[i]:
                        personal_best_positions[i], personal_best_scores[i] = random_search, func(random_search)
                        eval_count += 2

                indices = np.random.choice(np.delete(np.arange(self.swarm_size), i), 3, replace=False)
                x0, x1, x2 = positions[indices]
                mutated_vector = np.clip(x0 + self.mutation_factor * (x1 - x2), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob, mutated_vector, positions[i])
                
                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                if trial_score < global_best_score:
                    global_best_position = trial_vector
                    global_best_score = trial_score

                if eval_count >= self.budget:
                    break
            
            iteration += 1

        return global_best_position, global_best_score