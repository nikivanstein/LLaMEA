import numpy as np

class ChaoticQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.inertia_weight = 0.9
        self.cognitive_component = 2.0
        self.social_component = 2.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def chaotic_map(self, iteration):
        return 0.7 * (1 - (iteration / self.budget)) * np.sin(10 * np.pi * iteration / self.budget)

    def quantum_operator(self, particle, global_best, chaotic_factor):
        distance = np.linalg.norm(particle - global_best)
        random_vector = np.random.uniform(-1, 1, self.dim)
        return particle + chaotic_factor * distance * random_vector

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Evaluate the fitness of the particle
                fitness = func(self.positions[i])
                self.evaluations += 1

                # Update personal best
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                # Update global best
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i]

            chaotic_factor = self.chaotic_map(self.evaluations)

            # Update velocities and positions with quantum-inspired behavior
            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_component * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                quantum_move = self.quantum_operator(self.positions[i], self.global_best_position, chaotic_factor)
                self.positions[i] = np.clip(quantum_move, self.lower_bound, self.upper_bound)

            # Perform adaptive chaotic crossover
            if np.random.rand() < 0.2:  # Increased crossover probability
                for i in range(self.population_size):
                    crossover_point = np.random.randint(0, self.dim)
                    self.positions[i][:crossover_point] = chaotic_factor * self.global_best_position[:crossover_point] + \
                                                          (1 - chaotic_factor) * self.positions[i][:crossover_point]

        return self.global_best_score