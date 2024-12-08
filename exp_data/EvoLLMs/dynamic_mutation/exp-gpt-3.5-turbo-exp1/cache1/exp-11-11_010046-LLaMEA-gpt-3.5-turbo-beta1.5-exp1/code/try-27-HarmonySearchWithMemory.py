class HarmonySearchWithMemory:
    def __call__(self, func):
        for _ in range(self.budget):
            new_solution = np.clip(np.random.normal(np.mean(self.harmony_memory[:self.memory_size//2], axis=0), self.bandwidth), -5.0, 5.0)
            if func(new_solution) < func(self.harmony_memory[0]):
                self.harmony_memory[0] = new_solution
                self.harmony_memory = self.harmony_memory[self.harmony_memory[:,0].argsort()]
        return self.harmony_memory[0]