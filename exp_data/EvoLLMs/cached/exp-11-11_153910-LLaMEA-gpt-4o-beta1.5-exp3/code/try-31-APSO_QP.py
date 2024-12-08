import numpy as np

class APSO_QP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.evaluations = 0

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_perturbation(self, position):
        alpha = np.random.uniform(0, 1, self.dim)
        return position + alpha * np.sin(2 * np.pi * alpha)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.population[i])
            self.personal_best_fitness[i] = fitness
            self.personal_best_positions[i] = self.population[i]
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                
                # Update velocity
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.population[i]) +
                                      self.social_coeff * r2 * (self.global_best_position - self.population[i]))
                
                # Update position
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

                # Apply quantum perturbation
                perturbed_position = self.quantum_perturbation(self.population[i])
                perturbed_position = np.clip(perturbed_position, self.lower_bound, self.upper_bound)

                # Evaluate
                perturb_fitness = self.evaluate(func, perturbed_position)
                if perturb_fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = perturb_fitness
                    self.personal_best_positions[i] = perturbed_position
                    if perturb_fitness < self.global_best_fitness:
                        self.global_best_fitness = perturb_fitness
                        self.global_best_position = perturbed_position

        return self.global_best_position