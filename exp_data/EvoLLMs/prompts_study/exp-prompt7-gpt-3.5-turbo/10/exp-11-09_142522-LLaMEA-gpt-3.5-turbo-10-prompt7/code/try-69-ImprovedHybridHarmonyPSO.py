def dynamic_population_size_adjustment(pop_size, iteration):
    return int(pop_size * (1 - np.exp(-iteration / 100)))

class ImprovedHybridHarmonyPSO(HybridHarmonyPSO):
    def __call__(self, func):
        def initialize_harmony_memory(pop_size):
            return np.random.uniform(self.lower_bound, self.upper_bound, (pop_size, self.dim))

        pop_size = self.hm_size
        harmony_memory = initialize_harmony_memory(pop_size)
        fitness = np.array(self.evaluate_solutions_parallel(func, harmony_memory))
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for _ in range(self.budget - pop_size):
                new_solutions = []
                for _ in range(self.num_threads):
                    new_solution = np.zeros(self.dim)
                    hmcr = max(0.3, self.hmcr - 0.001)  # Dynamic adjustment of HMCR
                    par = min(0.6, self.par + 0.001)  # Dynamic adjustment of PAR
                    for i in range(self.dim):
                        if np.random.rand() < hmcr:
                            new_solution[i] = harmony_memory[np.random.choice(pop_size)][i]
                        else:
                            new_solution[i] = np.random.uniform(self.lower_bound, self.upper_bound)
                        new_solution[i] = new_solution[i] + np.random.uniform(-self.bandwidth, self.bandwidth)
                        new_solution[i] = np.clip(new_solution[i], self.lower_bound, self.upper_bound)
                    new_solutions.append(new_solution)
                
                # Evaluate new solutions in parallel
                new_fitness = np.array(self.evaluate_solutions_parallel(func, new_solutions))

                for idx, new_solution in enumerate(new_solutions):
                    worst_index = np.argmax(fitness)
                    if new_fitness[idx] < fitness[worst_index]:
                        harmony_memory[worst_index] = new_solution
                        fitness[worst_index] = new_fitness[idx]

                # Particle Swarm Optimization (PSO) for local exploration
                pso_swarm = np.random.uniform(self.lower_bound, self.upper_bound, (pop_size, self.dim))
                pbest = pso_swarm.copy()
                pbest_fitness = np.array(self.evaluate_solutions_parallel(func, pbest))
                gbest_index = np.argmin(pbest_fitness)
                gbest = pbest[gbest_index]

                for _ in range(self.pso_max_iter):
                    for i in range(pop_size):
                        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                        pso_velocity = self.pso_inertia * pso_swarm[i] + self.pso_c1 * r1 * (pbest[i] - pso_swarm[i]) + self.pso_c2 * r2 * (gbest - pso_swarm[i])
                        pso_swarm[i] = np.clip(pso_swarm[i] + pso_velocity, self.lower_bound, self.upper_bound)
                        pso_fitness = func(pso_swarm[i])
                        if pso_fitness < pbest_fitness[i]:
                            pbest[i] = pso_swarm[i]
                            pbest_fitness[i] = pso_fitness
                            if pbest_fitness[i] < func(gbest):
                                gbest = pbest[i]
                                gbest_index = i

                pop_size = dynamic_population_size_adjustment(pop_size, _)
                harmony_memory = np.concatenate((harmony_memory, initialize_harmony_memory(pop_size)), axis=0)
                fitness = np.concatenate((fitness, np.array(self.evaluate_solutions_parallel(func, initialize_harmony_memory(pop_size)))), axis=0)

        best_index = np.argmin(fitness)
        return harmony_memory[best_index], fitness[best_index]