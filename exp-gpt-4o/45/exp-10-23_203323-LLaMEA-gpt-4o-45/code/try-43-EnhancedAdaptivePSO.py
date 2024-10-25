import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 60  # Increased the number of particles further for better exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.85  # Adjusted maximum inertia weight for dynamic exploration-exploitation balance
        self.w_min = 0.2  # Reduced minimum inertia weight for more exploitation as convergence nears
        self.c1_initial = 1.8  # Adjusted cognitive factor to enhance exploration
        self.c2_initial = 1.4
        self.c1_final = 1.0
        self.c2_final = 3.0  # Further increased social factor to enhance convergence towards the global best
        self.velocity_clamp = 0.6  # Adjusted velocity clamping for dynamic control
        self.local_search_probability = 0.15  # Increased probability to perform local search with crossover strategy

    def __call__(self, func):
        np.random.seed(0)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_particles):
                score = func(positions[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = positions[i].copy()

                if evaluations >= self.budget:
                    break
            
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
            c1 = self.c1_initial - ((self.c1_initial - self.c1_final) * evaluations / self.budget)
            c2 = self.c2_initial + ((self.c2_final - self.c2_initial) * evaluations / self.budget)

            for i in range(self.num_particles):
                if np.random.rand() < self.local_search_probability:
                    crossover_partner = np.random.randint(0, self.num_particles)
                    crossover_position = 0.5 * (positions[i] + positions[crossover_partner])
                    crossover_position = np.clip(crossover_position, self.lower_bound, self.upper_bound)
                    crossover_score = func(crossover_position)
                    evaluations += 1
                    if crossover_score < personal_best_scores[i]:
                        personal_best_scores[i] = crossover_score
                        personal_best_positions[i] = crossover_position.copy()
                        if crossover_score < global_best_score:
                            global_best_score = crossover_score
                            global_best_position = crossover_position.copy()

                cognitive_component = c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = c2 * np.random.uniform(0, 1, self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                
                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)
                
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)