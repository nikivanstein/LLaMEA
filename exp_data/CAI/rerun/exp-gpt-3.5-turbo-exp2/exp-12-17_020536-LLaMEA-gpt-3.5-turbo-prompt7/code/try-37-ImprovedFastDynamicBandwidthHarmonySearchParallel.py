from concurrent.futures import ThreadPoolExecutor

class ImprovedFastDynamicBandwidthHarmonySearchParallel(FastDynamicBandwidthHarmonySearch):
    def __call__(self, func):
        pitch_rate = 0.45
        bandwidth = 0.1
        with ThreadPoolExecutor() as executor:
            for _ in range(self.budget):
                new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                for _ in range(2):
                    pitch_rate = max(0.01, pitch_rate * 0.97 * 1.3 * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))
                    for i in range(self.dim):
                        if np.random.rand() < pitch_rate:
                            new_harmony[i] = np.random.choice(self.harmony_memory[:, i])
                if func(new_harmony) < func(self.harmony_memory[-1]):
                    self.harmony_memory[-1] = new_harmony
                    self.harmony_memory = self.harmony_memory[np.argsort([func(h) for h in self.harmony_memory])]
                for _ in range(2):
                    local_search_results = list(executor.map(self._local_search, range(self.dim)))
                    best_local_harmony = min(local_search_results, key=lambda x: func(x))
                    if func(best_local_harmony) < func(self.harmony_memory[-1]):
                        self.harmony_memory[-1] = best_local_harmony
                bandwidth = min(0.5, max(0.01, bandwidth * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))))
        return self.harmony_memory[0]
    
    def _local_search(self, i):
        new_harmony_local = np.copy(self.harmony_memory[-1])
        new_harmony_local[i] = np.random.uniform(max(self.lower_bound, self.harmony_memory[-1][i] - bandwidth),
                                                 min(self.upper_bound, self.harmony_memory[-1][i] + bandwidth))
        return new_harmony_local