import numpy as np

class PSOGaussianOptimization:
    def __init__(self, budget, dim, swarm_size=30, omega=0.5, phi_p=0.5, phi_g=0.5, sigma=0.1):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.sigma = sigma

    def __call__(self, func):
        def gaussian_mutation(individual):
            return np.clip(individual + np.random.normal(0, self.sigma, self.dim), -5.0, 5.0)

        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        velocities = np.zeros_like(swarm)

        global_best = swarm[np.argmin([func(ind) for ind in swarm])]

        for _ in range(self.budget):
            new_swarm = np.zeros_like(swarm)

            for i in range(self.swarm_size):
                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)

                velocities[i] = self.omega * velocities[i] + self.phi_p * r_p * (swarm[i] - swarm[i]) + self.phi_g * r_g * (global_best - swarm[i])
                new_position = np.clip(swarm[i] + velocities[i], -5.0, 5.0)
                mutated_position = gaussian_mutation(new_position)

                if func(mutated_position) < func(swarm[i]):
                    new_swarm[i] = mutated_position
                else:
                    new_swarm[i] = swarm[i]

            swarm = new_swarm
            current_best = swarm[np.argmin([func(ind) for ind in swarm])]
            if func(current_best) < func(global_best):
                global_best = current_best

        return global_best