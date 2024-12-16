def chaotic_update(x, v, pbest, gbest, chaos_param):
    chaos = np.sin(chaos_param * x) * np.cos(chaos_param * v)
    return x + chaos * (pbest - x) + chaos * (gbest - x)

class EnhancedDynamicMutationOppositionBasedExplorationPSO_DE_Optimizer(DynamicMutationOppositionBasedExplorationPSO_DE_Optimizer):
    def __call__(self, func):
        chaos_param = np.random.uniform(0.1, 1.0)
        for _ in range(self.budget):
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            
            for i in range(self.swarm_size):
                mutation_factor = np.clip(np.random.normal(mutation_factors[i], 0.1), 0.1, 0.9)
                new_sol = de(population[i], pbest[[i, (i+1)%self.swarm_size, (i+2)%self.swarm_size]], mutation_factor)
                new_sol = chaotic_update(new_sol, velocities[i], pbest[i], gbest, chaos_param)
                
                new_score = evaluate(new_sol)
                if new_score < pbest_scores[i]:
                    pbest[i] = new_sol
                    pbest_scores[i] = new_score
                    if new_score < gbest_score:
                        gbest = new_sol
                        gbest_score = new_score
                      
                # Opposite solution
                
                mutation_factors[i] = mutation_factor
        return gbest