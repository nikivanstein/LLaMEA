import numpy as np

class FastDynamicInertiaWeightEnhancedSocialSwarmOptimization(DynamicInertiaWeightEnhancedSocialSwarmOptimization):
    def __call__(self, func):
        def _initialize_swarm():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        def _local_search(x, f):
            step_size = 1.0
            for _ in range(20): 
                x_new = x + step_size * np.random.normal(size=self.dim)
                if f(x_new) < f(x):
                    x = x_new
                    step_size *= 1.2  
                else:
                    step_size *= 0.8  
            return x

        swarm = _initialize_swarm()
        best_position = swarm[np.argmin([func(x) for x in swarm])]
        best_solution = best_position.copy()  # Elitist mechanism to store the best solution
        for _ in range(self.budget):
            diversity = np.mean(np.std(swarm, axis=0))
            new_swarm_size = max(5, int(self.swarm_size * (1 - diversity / 10.0)))  
            selected_indices = np.random.choice(range(self.swarm_size), new_swarm_size, replace=False)
            swarm = swarm[selected_indices]
            self.swarm_size = new_swarm_size
            for i in range(self.swarm_size):
                r_p = np.random.uniform(0, 1, size=self.dim)
                r_g = np.random.uniform(0, 1)
                inertia_weight = 0.3 + 0.7 * np.random.uniform()  
                swarm[i] = inertia_weight * swarm[i] + self.phi_p * r_p * (best_position - swarm[i]) + self.phi_g * r_g * (best_position - swarm[i])
                swarm[i] = _local_search(swarm[i], func)
                best_position = swarm[i] if func(swarm[i]) < func(best_position) else best_position
            best_solution = best_position if func(best_position) < func(best_solution) else best_solution  # Update the best solution found so far
        return best_solution