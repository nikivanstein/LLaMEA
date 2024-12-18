class ParallelImprovedAdaptiveSwarmEvolutionOptimization(ImprovedAdaptiveSwarmEvolutionOptimization):
    def __call__(self, func):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(func, ind) for ind in swarm]
            fitness_values = [future.result() for future in futures]
        for i in range(self.swarm_size):
            if fitness_values[i] < best_fitness:
                best_position = swarm[i]
                best_fitness = fitness_values[i]
        return best_position