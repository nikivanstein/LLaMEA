import numpy as np

class AdaptiveMemeticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))
        self.omega = 0.5  # inertia weight
        self.phi_p = 1.5  # cognitive parameter
        self.phi_g = 1.5  # social parameter
        self.local_search_prob = 0.1  # probability of applying local search

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        num_evaluations = self.population_size

        def local_search(individual):
            """A simple local search strategy: perturb and evaluate."""
            candidate = individual + np.random.uniform(-0.1, 0.1, self.dim)
            candidate = np.clip(candidate, self.lb, self.ub)
            candidate_fitness = func(candidate)
            return candidate, candidate_fitness

        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break

                # Update velocities and positions
                r_p, r_g = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.omega * velocities[i] +
                                 self.phi_p * r_p * (personal_best_positions[i] - population[i]) +
                                 self.phi_g * r_g * (global_best_position - population[i]))
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lb, self.ub)

                # Evaluate new position
                fitness = func(population[i])
                num_evaluations += 1

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = population[i]

                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = population[i]

                # Perform local search with a certain probability
                if np.random.rand() < self.local_search_prob:
                    candidate, candidate_fitness = local_search(population[i])
                    num_evaluations += 1
                    if candidate_fitness < personal_best_fitness[i]:
                        personal_best_fitness[i] = candidate_fitness
                        personal_best_positions[i] = candidate
                        if candidate_fitness < global_best_fitness:
                            global_best_fitness = candidate_fitness
                            global_best_position = candidate

        return global_best_position, global_best_fitness