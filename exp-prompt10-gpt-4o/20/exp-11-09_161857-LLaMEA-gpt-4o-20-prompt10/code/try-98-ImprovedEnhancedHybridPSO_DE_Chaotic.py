import numpy as np

class ImprovedEnhancedHybridPSO_DE_Chaotic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5  # Adjusted cognitive coefficient for balanced personal attraction
        self.c2 = 1.5  # Adjusted social coefficient for balanced global attraction
        self.w = 0.5  # Lower inertia weight for faster convergence
        self.F_base = 0.5  # Adjusted mutation scaling factor for enhanced mutation
        self.CR = 0.9  # Higher crossover rate for better offspring diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        # Updated Chaotic Initialization using logistic map for better initial diversity
        self.positions = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.population_size, self.dim) * (np.sin(np.arange(self.population_size)[:, None]) + 0.5)
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.adaptive_w_factor = 0.98  # Adaptive inertia weight for gradual convergence control
        self.mutation_adjustment = 0.40  # Enhanced parameter for adaptive mutation
        self.diversity_threshold = 0.3  # Adjusted diversity threshold for mutation

    def __call__(self, func):
        function_evaluations = 0

        while function_evaluations < self.budget:
            # Evaluate current population
            scores = np.apply_along_axis(func, 1, self.positions)
            function_evaluations += self.population_size
            
            # Update personal bests and global best
            for i in range(self.population_size):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]
                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]
            
            # PSO Update with adaptive velocity strategy
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.personal_best_positions - self.positions) +
                               self.c2 * r2 * (self.global_best_position - self.positions))
            self.positions = self.positions + self.velocities
            self.w *= self.adaptive_w_factor  # Adaptive inertia weight decay
            
            # Differential Evolution Mutation and Crossover with adaptive mutation scaling
            diversity = np.mean(np.std(self.positions, axis=0))
            for i in range(self.population_size):
                if diversity < self.diversity_threshold:
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    F_adaptive = self.F_base + self.mutation_adjustment * (1 - (scores[i] / self.global_best_score))
                    mutant = self.positions[a] + F_adaptive * (self.positions[b] - self.positions[c])
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                    cross_points = np.random.rand(self.dim) < self.CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True

                    trial = np.where(cross_points, mutant, self.positions[i])

                    trial_score = func(trial)
                    function_evaluations += 1

                    if trial_score < scores[i]:
                        self.positions[i] = trial
                        scores[i] = trial_score
                        if trial_score < self.personal_best_scores[i]:
                            self.personal_best_scores[i] = trial_score
                            self.personal_best_positions[i] = trial

                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial

            # Ensure positions are within bounds
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

        return self.global_best_position