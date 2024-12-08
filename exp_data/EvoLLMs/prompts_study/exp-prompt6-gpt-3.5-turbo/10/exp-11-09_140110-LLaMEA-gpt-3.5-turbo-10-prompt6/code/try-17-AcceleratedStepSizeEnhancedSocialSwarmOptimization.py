import numpy as np

class AcceleratedStepSizeEnhancedSocialSwarmOptimization(DynamicPopulationSizeEnhancedSocialSwarmOptimization):
    def __call__(self, func):
        def _initialize_swarm():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        def _local_search(x, f):
            step_size = 1.0
            for _ in range(10):
                gradient = np.gradient(f(x))
                x_new = x - step_size * gradient  # Use gradient for local search
                if f(x_new) < f(x):
                    x = x_new
                    step_size *= 1.1
                else:
                    step_size *= 0.9
            return x

        swarm = _initialize_swarm()
        best_position = swarm[np.argmin([func(x) for x in swarm])]
        for _ in range(self.budget):
            diversity = np.mean(np.std(swarm, axis=0))
            new_swarm_size = max(5, int(self.swarm_size * (1 - diversity / 10.0)))
            selected_indices = np.random.choice(range(self.swarm_size), new_swarm_size, replace=False)
            swarm = swarm[selected_indices]
            self.swarm_size = new_swarm_size
            for i in range(self.swarm_size):
                r_p = np.random.uniform(0, 1, size=self.dim)
                r_g = np.random.uniform(0, 1)
                swarm[i] = self.omega * swarm[i] + self.phi_p * r_p * (best_position - swarm[i]) + self.phi_g * r_g * (best_position - swarm[i])
                swarm[i] = _local_search(swarm[i], func)
                best_position = swarm[i] if func(swarm[i]) < func(best_position) else best_position
        return best_position