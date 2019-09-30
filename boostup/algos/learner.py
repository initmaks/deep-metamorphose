class Learner():
    def __init__(self,):
        pass

    def var_count(self,):
        raise NotImplementedError

    def train(self,):
        raise NotImplementedError

    def eval(self,):
        raise NotImplementedError

    def learn(self, buffer):
        raise NotImplementedError

    def get_action(self,obs, deterministic):
        raise NotImplementedError

    def checkpoint(self):
        raise NotImplementedError