class PortfolioManager:
    def allocate(self,workers): return {w.name:1/len(workers) for w in workers}