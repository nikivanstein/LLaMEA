import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim, num_particles=30, max_iterations=1000, inertia_weight=0.7, cognitive_weight=1.4, social_weight=1.4, initial_temperature=100, cooling_rate=0.95):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def __call__(self, func):
        def pso_sa_helper():
            # PSO initialization
            best_particle_position = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
            best_particle_fitness = np.full(self.num_particles, np.inf)
            global_best_position = np.zeros(self.dim)
            global_best_fitness = np.inf

            # PSO optimization loop
            for _ in range(self.max_iterations):
                # PSO update rules

            # SA initialization
            current_solution = global_best_position
            current_fitness = func(global_best_position)

            # SA optimization loop
            for _ in range(self.max_iterations):
                proposed_solution = current_solution + np.random.normal(0, 1, self.dim)
                proposed_solution = np.clip(proposed_solution, -5.0, 5.0)
                proposed_fitness = func(proposed_solution)
                if acceptance_probability(current_fitness, proposed_fitness, self.initial_temperature) > np.random.rand():
                    current_solution = proposed_solution
                    current_fitness = proposed_fitness
                    if current_fitness < global_best_fitness:
                        global_best_position = current_solution
                        global_best_fitness = current_fitness

        return pso_sa_helper()

def acceptance_probability(current_fitness, proposed_fitness, temperature):
    if proposed_fitness < current_fitness:
        return 1.0
    return np.exp((current_fitness - proposed_fitness) / temperature)