import numpy as np

class AdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.population_size = self.initial_population_size
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.w_start = 0.9
        self.w_end = 0.4
        self.F_start = 0.5
        self.F_end = 0.9
        self.CR_start = 0.8
        self.CR_end = 0.95
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_evals = 0
        self.local_search_prob_start = 0.1
        self.local_search_prob_end = 0.3

    def __call__(self, func):
        def adapt_parameters():
            progress = self.num_evals / self.budget
            w = self.w_start - progress * (self.w_start - self.w_end)
            F = self.F_start + progress * (self.F_end - self.F_start)
            CR = self.CR_start + progress * (self.CR_end - self.CR_start)
            local_search_prob = self.local_search_prob_start + progress * (self.local_search_prob_end - self.local_search_prob_start)
            return w, F, CR, local_search_prob

        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        initial_velocity_clamp = (self.upper_bound - self.lower_bound) / 2.0

        def switch_phase(iteration):
            return (iteration < 0.5 * self.budget)

        def adjust_population_size(progress):
            self.population_size = int(self.initial_population_size * (1.0 - 0.5 * progress))

        while self.num_evals < self.budget:
            w, F, CR, local_search_prob = adapt_parameters()
            velocity_clamp = initial_velocity_clamp * (1 - self.num_evals / self.budget)
            exploration_phase = switch_phase(self.num_evals)
            adjust_population_size(self.num_evals / self.budget)

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

            for i in range(self.population_size):
                inertia = w * velocities[i]
                cognitive = self.c1 * np.random.random(self.dim) * (personal_best_positions[i] - swarm[i])
                social = self.c2 * np.random.random(self.dim) * (global_best_position - swarm[i])
                velocities[i] = inertia + cognitive + social
                velocities[i] = np.clip(velocities[i], -velocity_clamp, velocity_clamp)

                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if self.num_evals >= self.budget:
                    break
                if exploration_phase:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = personal_best_positions[indices]
                else:
                    candidates = np.random.choice(self.population_size, 5, replace=False)
                    best_candidate = min(candidates, key=lambda idx: personal_best_scores[idx])
                    a, b, c = personal_best_positions[[best_candidate] + list(np.random.choice(candidates, 2, replace=False))]

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

                if not exploration_phase and np.random.rand() < local_search_prob:
                    local_variance = 0.1 * (1 - (self.num_evals / self.budget))
                    local_trial = trial + np.random.normal(0, local_variance, self.dim)
                    local_trial = np.clip(local_trial, self.lower_bound, self.upper_bound)
                    local_trial_score = func(local_trial)
                    self.num_evals += 1

                    if local_trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = local_trial_score
                        personal_best_positions[i] = local_trial
                        if local_trial_score < global_best_score:
                            global_best_score = local_trial_score
                            global_best_position = local_trial

        return global_best_position