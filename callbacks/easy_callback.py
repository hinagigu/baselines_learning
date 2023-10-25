from stable_baselines3.common.callbacks import BaseCallback
class EasyCallBack(BaseCallback):
    def __init__(self):
        super().__init__()
        self.call_count = 0
    def _on_step(self):
        self.call_count += 1
        print(self.call_count)
        if self.call_count>=1000:
            return False
        return True