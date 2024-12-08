import numpy as np

class Hybrid_PSO_SA_CLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # PSO parameters
        self.num_particles = 40
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5

        # Simulated Annealing parameters
        self.initial_temp = 1000.0
        self.alpha = 0.99

        # Initialize particles and velocities
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def chaotic_lattice(self, x):
        return np.cos(np.pi * x)  # Chaotic map for lattice search

    def simulated_annealing(self, score, new_score, temperature):
        if new_score < score:
            return True
        return np.exp((score - new_score) / temperature) > np.random.rand()

    def __call__(self, func):
        evals = 0
        temperature = self.initial_temp

        while evals < self.budget:
            # Evaluate each particle
            scores = np.apply_along_axis(func, 1, self.positions)
            evals += self.num_particles

            # Update personal and global bests
            for i in range(self.num_particles):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            # Update velocities and positions (PSO)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Perform Simulated Annealing with Chaotic Lattice Search
            for i in range(self.num_particles):
                candidate_position = self.positions[i] + self.chaotic_lattice(np.random.rand(self.dim))
                candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                candidate_score = func(candidate_position)

                if self.simulated_annealing(scores[i], candidate_score, temperature):
                    self.positions[i] = candidate_position
                    scores[i] = candidate_score

            temperature *= self.alpha  # Reduce temperature for simulated annealing

        return self.global_best_position, self.global_best_score