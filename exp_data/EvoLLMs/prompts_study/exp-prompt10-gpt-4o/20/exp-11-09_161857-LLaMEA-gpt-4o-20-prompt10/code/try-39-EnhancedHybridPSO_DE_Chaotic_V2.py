import numpy as np

class EnhancedHybridPSO_DE_Chaotic_V2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60  # Slightly increased population for better diversity
        self.c1 = 1.7  # Adjusted cognitive coefficient
        self.c2 = 1.3  # Adjusted social coefficient for balanced exploration and exploitation
        self.w = 0.7  # Adjusted inertia weight
        self.F_base = 0.5  # Increased base mutation factor
        self.CR = 0.85  # Modified crossover rate for more trial solutions
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        # Enhanced Chaotic Initialization for better coverage
        self.positions = self.lower_bound + (self.upper_bound - self.lower_bound) * (np.random.rand(self.population_size, self.dim) - 0.5) * np.sin(np.arange(self.population_size)[:, None] * np.pi / 2)
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.adaptive_w_factor = 0.98  # Adjusted adaptive factor for inertia weight
        self.mutation_adjustment = 0.4  # Enhanced parameter for adaptive mutation

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
            
            # PSO Update with adaptive velocity damping
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.personal_best_positions - self.positions) +
                               self.c2 * r2 * (self.global_best_position - self.positions))
            self.positions = self.positions + self.velocities
            self.w *= self.adaptive_w_factor  # Apply adaptive inertia weight
            
            # Differential Evolution Mutation and Crossover with adaptive mutation scaling
            diversity = np.mean(np.std(self.positions, axis=0))
            for i in range(self.population_size):
                if diversity < 0.15:  # Optimized threshold for diversity check
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

            # Ensure positions are within bounds and introduce chaotic perturbations
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)
            chaotic_perturbation = np.sin(function_evaluations * np.pi / self.budget)
            self.positions += chaotic_perturbation * np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))

        return self.global_best_position