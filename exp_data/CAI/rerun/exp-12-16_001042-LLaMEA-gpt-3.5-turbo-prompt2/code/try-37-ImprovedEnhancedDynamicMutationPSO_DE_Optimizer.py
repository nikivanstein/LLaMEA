class ImprovedEnhancedDynamicMutationPSO_DE_Optimizer(EnhancedDynamicMutationPSO_DE_Optimizer):
    def __call__(self, func):
        def chaotic_local_search(x, pbest, gbest, mutation_factor):
            chaotic_factor = np.random.uniform(0.5, 1.5, size=self.dim)
            chaotic_sol = x + chaotic_factor * np.random.rand(self.dim) * (pbest - x) + chaotic_factor * np.random.rand(self.dim) * (gbest - x)
            return chaotic_sol

        def evaluate(x):
            return func(x)

        population = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        pbest = population.copy()
        pbest_scores = np.array([evaluate(p) for p in pbest])
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx]
        gbest_score = pbest_scores[gbest_idx]

        for _ in range(self.budget):
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            population += velocities

            for i in range(self.swarm_size):
                mutation_factor = np.clip(np.random.normal(self.mutation_factor, 0.1), 0.1, 0.9)
                new_sol = de(population[i], pbest[[i, (i+1)%self.swarm_size, (i+2)%self.swarm_size]], mutation_factor)
                new_score = evaluate(new_sol)
                if new_score < pbest_scores[i]:
                    pbest[i] = new_sol
                    pbest_scores[i] = new_score
                    if new_score < gbest_score:
                        gbest = new_sol
                        gbest_score = new_score

                if np.random.rand() < 0.1:
                    dynamic_mutant = np.mean(np.array([pbest[i], gbest]), axis=0) + np.random.uniform(-1, 1, size=self.dim)
                    dynamic_score = evaluate(dynamic_mutant)
                    if dynamic_score < pbest_scores[i]:
                        pbest[i] = dynamic_mutant
                        pbest_scores[i] = dynamic_score
                        if dynamic_score < gbest_score:
                            gbest = dynamic_mutant
                            gbest_score = dynamic_score

                # Introducing chaotic local search for more intensive exploration
                if np.random.rand() < 0.05:
                    chaotic_sol = chaotic_local_search(population[i], pbest[i], gbest, mutation_factor)
                    chaotic_score = evaluate(chaotic_sol)
                    if chaotic_score < pbest_scores[i]:
                        pbest[i] = chaotic_sol
                        pbest_scores[i] = chaotic_score
                        if chaotic_score < gbest_score:
                            gbest = chaotic_sol
                            gbest_score = chaotic_score

        return gbest