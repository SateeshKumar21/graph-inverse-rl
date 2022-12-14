import os
import numpy as np

class StateRecorder(object):
    def __init__(self, dir_name, height=256, width=256, fps=30):
        self.dir_name = dir_name
        self.save_dir = dir_name if dir_name else None
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
       
    def record(self, state):
        if self.enabled:
            frame = state
            
            self.frames.append(frame)
            # frame = env.render(mode="rgb_array")
            # self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            print(f"Shape of frames {len(self.frames)}")
            np.save(path, self.frames)
            #imageio.mimsave(path, self.frames, fps=self.fps