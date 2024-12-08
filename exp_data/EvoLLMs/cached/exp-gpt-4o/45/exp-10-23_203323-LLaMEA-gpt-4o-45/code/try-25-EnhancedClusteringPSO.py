import numpy as np

class EnhancedClusteringPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 60  # Increased particle count for better initial coverage
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.8
        self.w_min = 0.2  # Further reduced to enhance exploitation in later stages
        self.c1_initial = 1.8  # Fine-tuned to enhance personal exploration
        self.c2_initial = 1.7
        self.c1_final = 1.0
        self.c2_final = 2.9  # Further increase for stronger global convergence
        self.velocity_clamp = 0.6  # Improved dynamic control
        self.local_search_probability = 0.2  # Increased probability for more frequent local exploration

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
            # Dynamic clustering for diverse exploration
            if evaluations % (self.budget // 10) == 0: 
                clusters = np.random.randint(0, 5, self.num_particles)  # Randomly assign particles to clusters

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
                    perturbation = np.random.normal(0, 0.2, self.dim)
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

                # Introduce clustering effect on velocity update
                cluster_center = np.mean(positions[clusters == clusters[i]], axis=0)
                cognitive_component = c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = c2 * np.random.uniform(0, 1, self.dim) * (global_best_position - positions[i])
                cluster_component = 0.1 * (cluster_center - positions[i])  # Small influence from the cluster center

                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component + cluster_component
                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)
                
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

# Usage:
# enhanced_pso = EnhancedClusteringPSO(budget=10000, dim=10)
# enhanced_pso(func)