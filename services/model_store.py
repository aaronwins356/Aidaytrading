import joblib,os
class ModelStore:
    def __init__(self,path='models'): os.makedirs(path,exist_ok=True); self.path=path
    def load(self,name):
        p=f'{self.path}/{name}.pkl'
        return joblib.load(p) if os.path.exists(p) else None
    def save(self,name,model): joblib.dump(model,f'{self.path}/{name}.pkl')