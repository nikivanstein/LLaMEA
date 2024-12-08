import numpy as np

class DynamicTopologyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 60  # Increased number of particles for broader exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.8
        self.w_min = 0.4  # Adjusted inertia weight for better balance
        self.c1_initial = 1.8  # Fine-tuned cognitive factor
        self.c2_initial = 1.7
        self.c1_final = 1.0
        self.c2_final = 2.5  # Reduced social factor for more balanced convergence
        self.velocity_clamp = 0.6  # Adjusted velocity clamping for dynamic control
        self.local_search_probability = 0.15  # Increased probability for local search
        self.topology_switch_probability = 0.2  # Probability to switch between topologies

    def __call__(self, func):
        np.random.seed(0)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        use_ring_topology = True  # Start with a ring topology

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

            if np.random.rand() < self.topology_switch_probability:
                use_ring_topology = not use_ring_topology  # Switch topology

            for i in range(self.num_particles):
                if np.random.rand() < self.local_search_probability:
                    perturbation = np.random.normal(0, 0.15, self.dim)  # Increased mutation strength
                    candidate_position = positions[i] + perturbation
                    candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                    candidate_score = func(candidate_position)
                    evaluations += 1
                    if candidate_score < personal_best_scores[i]:
                        personal_best_scores[i] = candidate_score
                        personal_best_positions[i] = candidate_position.copy()
                        if candidate_score < global_best_score:
                            global_best_score = candidate_score
                            global_best_position = candidate_position.copy()

                if use_ring_topology:
                    neighbors = [personal_best_positions[j] for j in [(i-1) % self.num_particles, i, (i+1) % self.num_particles]]
                    best_neighbor_position = min(neighbors, key=lambda x: func(x))
                else:
                    best_neighbor_position = global_best_position

                cognitive_component = c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = c2 * np.random.uniform(0, 1, self.dim) * (best_neighbor_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component

                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)