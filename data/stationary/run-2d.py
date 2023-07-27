from lotus import run
from pathlib import Path
import time


def run2d():
    run(64, f'{cwd}/lotus-data')

if __name__ == "__main__":
    cwd = Path.cwd()
    run2d()
    # time.sleep(60*5)

    

