import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.49618  # Cognitive component
        self.c2 = 1.49618  # Social component
        self.w_start = 0.9  # Start inertia weight
        self.w_end = 0.4    # End inertia weight
        self.F_start = 0.5
        self.F_end = 0.9
        self.CR_start = 0.8
        self.CR_end = 0.95
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_evals = 0
        self.niche_radius = 0.1  # Dynamic niching parameter

    def __call__(self, func):
        def adapt_parameters():
            progress = self.num_evals / self.budget
            w = self.w_start - progress * (self.w_start - self.w_end)
            F = self.F_start + progress * (self.F_end - self.F_start)
            CR = self.CR_start + progress * (self.CR_end - self.CR_start)
            return w, F, CR

        # Initialize swarm
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        initial_velocity_clamp = (self.upper_bound - self.lower_bound) / 2.0

        # Introduce a phase-based approach with adaptive strategies
        def switch_phase(iteration):
            return (iteration < 0.3 * self.budget or iteration > 0.7 * self.budget)

        while self.num_evals < self.budget:
            w, F, CR = adapt_parameters()
            velocity_clamp = initial_velocity_clamp * (1 - self.num_evals / self.budget)
            exploration_phase = switch_phase(self.num_evals)

            # Evaluate current swarm
            for i in range(self.population_size):
                if self.num_evals >= self.budget:
                    break
                score = func(swarm[i])
                self.num_evals += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm[i]

            # PSO update with adaptive inertia and dynamic velocity clamping
            for i in range(self.population_size):
                inertia = w * velocities[i]
                cognitive = self.c1 * np.random.random(self.dim) * (personal_best_positions[i] - swarm[i])
                social = self.c2 * np.random.random(self.dim) * (global_best_position - swarm[i])
                velocities[i] = inertia + cognitive + social
                velocities[i] = np.clip(velocities[i], -velocity_clamp, velocity_clamp)

                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

            # DE-inspired mutation and crossover with dynamic niching in exploration phase
            for i in range(self.population_size):
                if self.num_evals >= self.budget:
                    break
                if exploration_phase:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = personal_best_positions[indices]
                    mutant = a + F * (b - c)
                else:
                    distances = np.linalg.norm(swarm - swarm[i], axis=1)
                    niche_candidates = np.where(distances < self.niche_radius)[0]
                    if len(niche_candidates) > 2:
                        a, b, c = personal_best_positions[np.random.choice(niche_candidates, 3, replace=False)]
                    else:
                        a, b, c = personal_best_positions[np.random.choice(self.population_size, 3, replace=False)]
                    mutant = a + F * (b - c)

                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                trial = np.copy(swarm[i])
                for j in range(self.dim):
                    if np.random.rand() < CR or j == np.random.randint(0, self.dim):
                        trial[j] = mutant[j]

                trial_score = func(trial)
                self.num_evals += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial

        return global_best_position