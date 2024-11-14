class ImprovedEvolutionaryHarmonySearch(EvolutionaryHarmonySearch):
    def __call__(self, func):
        def explore_new_solution(harmony_memory):
            new_solution = np.clip(harmony_memory[np.random.randint(self.hm_size)] + np.random.uniform(-self.bw, self.bw, self.dim),
                                    -5.0, 5.0)
            return new_solution