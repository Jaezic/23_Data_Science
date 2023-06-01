import datetime
import os
import numpy as np
import random

# makes the random numbers predictable
def set_seed(rand_seed):
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H_%M_%S'

    #     time.strftime(format[, t])
    return datetime.datetime.today().strftime(fmt)

class ReDirectSTD(object):
    """
    overwrites the sys.stdout or sys.stderr
    Args:
      fpath: file cam_path
      console: one of ['stdout', 'stderr']
      immediately_visiable: False
    Usage example:
      ReDirectSTD('stdout.txt', 'stdout', False)
      ReDirectSTD('stderr.txt', 'stderr', False)
    """

    def __init__(self, fpath=None, console='stdout', immediately_visiable=False):
        import sys
        import os
        assert console in ['stdout', 'stderr']
        self.console = sys.stdout if console == "stdout" else sys.stderr
        self.file = fpath
        self.f = None
        self.immediately_visiable = immediately_visiable

        if fpath is not None:
            # Remove existing log file
            if os.path.exists(fpath):
                os.remove(fpath)
        if console == 'stdout':
            sys.stdout = self
        else:
            sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, **args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            if not os.path.exists(os.path.dirname(os.path.abspath(self.file))):
                os.mkdir(os.path.dirname(os.path.abspath(self.file)))

            if self.immediately_visiable:
                # open for writing, appending to the end of the file if it exists
                with open(self.file, 'a') as f:
                    f.write(msg)
            else:
                if self.f is None:
                    self.f = open(self.file, 'w')

                # print("self.f is not none")
                # first time self.f is None, second is not None
                self.f.write(msg)

    def flush(self):
        self.console.flush()
        if self.f is not None:
            self.f.flush()
            import os
            os.fsync(self.f.fileno())

    def close(self):
        self.console.close()
        if self.f is not None:
            self.f.close()
