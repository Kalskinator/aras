from tqdm import tqdm

class ProgressBarHelper:
    def __init__(self, total, desc):
        self.progress_bar = tqdm(total=total, desc=desc, leave=False)

    def update(self, n=1):
        self.progress_bar.update(n)

    def close(self):
        self.progress_bar.close()