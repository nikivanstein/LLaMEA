import numpy as np

class MultiSwarmCooperativePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 2
        self.num_particles_per_swarm = 25  # Divide particles across multiple swarms
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.9
        self.w_min = 0.4  # Adjusted for better balance between exploration and exploitation
        self.c1_initial = 1.8
        self.c2_initial = 2.0
        self.c1_final = 1.0
        self.c2_final = 3.0  # Increased social factor for stronger convergence
        self.velocity_clamp = 0.8  # Adjusted velocity clamping
        self.local_search_probability = 0.15  # Increased local search probability
        self.chaotic_search_probability = 0.05  # Introduced chaotic search

    def __call__(self, func):
        np.random.seed(0)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_swarms, self.num_particles_per_swarm, self.dim))
        velocities = np.zeros((self.num_swarms, self.num_particles_per_swarm, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full((self.num_swarms, self.num_particles_per_swarm), float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        while evaluations < self.budget:
            for swarm in range(self.num_swarms):
                for i in range(self.num_particles_per_swarm):
                    score = func(positions[swarm][i])
                    evaluations += 1
                    if score < personal_best_scores[swarm][i]:
                        personal_best_scores[swarm][i] = score
                        personal_best_positions[swarm][i] = positions[swarm][i].copy()
                        if score < global_best_score:
                            global_best_score = score
                            global_best_position = positions[swarm][i].copy()

                    if evaluations >= self.budget:
                        break
                
                if evaluations >= self.budget:
                    break

                inertia_weight = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
                c1 = self.c1_initial - ((self.c1_initial - self.c1_final) * evaluations / self.budget)
                c2 = self.c2_initial + ((self.c2_final - self.c2_initial) * evaluations / self.budget)

                for i in range(self.num_particles_per_swarm):
                    if np.random.rand() < self.local_search_probability:
                        perturbation = np.random.normal(0, 0.1, self.dim)
                        candidate_position = positions[swarm][i] + perturbation
                        candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                        candidate_score = func(candidate_position)
                        evaluations += 1
                        if candidate_score < personal_best_scores[swarm][i]:
                            personal_best_scores[swarm][i] = candidate_score
                            personal_best_positions[swarm][i] = candidate_position.copy()
                            if candidate_score < global_best_score:
                                global_best_score = candidate_score
                                global_best_position = candidate_position.copy()

                    if np.random.rand() < self.chaotic_search_probability:
                        # Perform chaotic search
                        chaos_factor = np.random.uniform(-0.2, 0.2, self.dim)
                        chaotic_position = positions[swarm][i] + chaos_factor
                        chaotic_position = np.clip(chaotic_position, self.lower_bound, self.upper_bound)
                        chaotic_score = func(chaotic_position)
                        evaluations += 1
                        if chaotic_score < personal_best_scores[swarm][i]:
                            personal_best_scores[swarm][i] = chaotic_score
                            personal_best_positions[swarm][i] = chaotic_position.copy()
                            if chaotic_score < global_best_score:
                                global_best_score = chaotic_score
                                global_best_position = chaotic_position.copy()

                    cognitive_component = c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[swarm][i] - positions[swarm][i])
                    social_component = c2 * np.random.uniform(0, 1, self.dim) * (global_best_position - positions[swarm][i])
                    velocities[swarm][i] = inertia_weight * velocities[swarm][i] + cognitive_component + social_component
                    
                    velocities[swarm][i] = np.clip(velocities[swarm][i], -self.velocity_clamp, self.velocity_clamp)
                    
                    positions[swarm][i] += velocities[swarm][i]
                    positions[swarm][i] = np.clip(positions[swarm][i], self.lower_bound, self.upper_bound)

# Usage:
# multi_swarm_pso = MultiSwarmCooperativePSO(budget=10000, dim=10)
# multi_swarm_pso(func)