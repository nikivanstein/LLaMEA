import numpy as np

class EnhancedHybridPSO_DE_Chaotic_MultiSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.num_swarms = 2  # Introduce multiple swarms
        self.c1 = 1.5  # Reduced cognitive coefficient for balanced personal and social influences
        self.c2 = 1.5  # Increased social coefficient for diversified search
        self.w = 0.7  # Slightly increased inertia for better exploration early on
        self.F_base = 0.5  # Increased mutation factor for robust differential evolution
        self.CR = 0.9  # Increased crossover rate for better mixing
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        self.positions = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.population_size, self.dim)
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.adaptive_w_factor = 0.97  # Slightly adjusted adaptive factor for inertia
        self.mutation_adjustment = 0.4  # Adjusted parameter for adaptive mutation

    def __call__(self, func):
        function_evaluations = 0
        
        while function_evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, self.positions)
            function_evaluations += self.population_size

            for i in range(self.population_size):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]
                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            for swarm_id in range(self.num_swarms):
                r1 = np.random.rand(self.population_size // self.num_swarms, self.dim)
                r2 = np.random.rand(self.population_size // self.num_swarms, self.dim)
                swarm_slice = slice(swarm_id * (self.population_size // self.num_swarms),
                                    (swarm_id + 1) * (self.population_size // self.num_swarms))
                self.velocities[swarm_slice] = (self.w * self.velocities[swarm_slice] +
                                                self.c1 * r1 * (self.personal_best_positions[swarm_slice] - self.positions[swarm_slice]) +
                                                self.c2 * r2 * (self.global_best_position - self.positions[swarm_slice]))
                self.positions[swarm_slice] += self.velocities[swarm_slice]
                self.w *= self.adaptive_w_factor

            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                F_adaptive = self.F_base + self.mutation_adjustment * (1 - (scores[i] / (self.global_best_score + 1e-9)))
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

            # Local search around global best for refined exploitation
            if function_evaluations < self.budget:
                local_trial = self.global_best_position + 0.05 * np.random.randn(self.dim)
                local_trial = np.clip(local_trial, self.lower_bound, self.upper_bound)
                local_score = func(local_trial)
                function_evaluations += 1
                if local_score < self.global_best_score:
                    self.global_best_score = local_score
                    self.global_best_position = local_trial

            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

        return self.global_best_position