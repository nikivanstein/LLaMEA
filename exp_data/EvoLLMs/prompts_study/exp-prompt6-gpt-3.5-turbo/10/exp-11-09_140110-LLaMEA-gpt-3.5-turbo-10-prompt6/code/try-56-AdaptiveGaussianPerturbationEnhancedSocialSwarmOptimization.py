import numpy as np

class AdaptiveGaussianPerturbationEnhancedSocialSwarmOptimization(DynamicInertiaWeightEnhancedSocialSwarmOptimization):
    def __call__(self, func):
        def _initialize_swarm():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        def _adaptive_gaussian_perturbation_mutation(x, f):
            step_size = 1.0
            for _ in range(10):
                perturbation = np.random.normal(scale=step_size, size=self.dim)
                x_new = x + perturbation
                if f(x_new) < f(x):
                    x = x_new
                    step_size *= 1.1  # Adaptive step adjustment
                else:
                    step_size *= 0.9  # Adaptive step adjustment
            return x

        swarm = _initialize_swarm()
        best_position = swarm[np.argmin([func(x) for x in swarm])]
        for _ in range(self.budget):
            diversity = np.mean(np.std(swarm, axis=0))
            new_swarm_size = max(5, int(self.swarm_size * (1 - diversity / 10.0)))  # Dynamic population size adjustment
            selected_indices = np.random.choice(range(self.swarm_size), new_swarm_size, replace=False)
            swarm = swarm[selected_indices]
            self.swarm_size = new_swarm_size
            for i in range(self.swarm_size):
                r_p = np.random.uniform(0, 1, size=self.dim)
                r_g = np.random.uniform(0, 1)
                inertia_weight = 0.5 + 0.5 * np.random.uniform()  # Dynamic inertia weight adaptation
                swarm[i] = inertia_weight * swarm[i] + self.phi_p * r_p * (best_position - swarm[i]) + self.phi_g * r_g * (best_position - swarm[i])
                swarm[i] = _adaptive_gaussian_perturbation_mutation(swarm[i], func)
                best_position = swarm[i] if func(swarm[i]) < func(best_position) else best_position
        return best_position