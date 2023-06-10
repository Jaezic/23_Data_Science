import datetime
import os
import numpy as np
import random
import pandas as pd

# makes the random numbers predictable
def set_seed(rand_seed):
    """
    Set random seed
        Args:
            rand_seed: random seed
        
        Returns:
            None
    """
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)


def time_str(fmt=None):
    """
    Source : https://docs.python.org/ko/3/library/datetime.html
    Get time string
        Args:
            fmt: format of the time string
        
        Returns:
            time string
    """
    if fmt is None:
        fmt = '%Y-%m-%d_%H_%M_%S'

    #     time.strftime(format[, t])
    return datetime.datetime.today().strftime(fmt)


# Source : https://stackoverflow.com/questions/6796492/temporarily-redirect-stdout-stderr
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
    """
    What is ReDirectSTD?
    ReDirectSTD is a class that overwrites the sys.stdout or sys.stderr.
    It is used to save the log of the program to a file.
    
    How to use ReDirectSTD?
    Usage example:
        ReDirectSTD('stdout.txt', 'stdout', False)
        
        Run the program, and the log will be saved to the file 'stdout.txt'.
    """
    def __init__(self, fpath=None, console='stdout', immediately_visiable=False):
        """
        overwrites the sys.stdout or sys.stderr
        Args:
            fpath: file cam_path
            console: one of ['stdout', 'stderr']
            immediately_visiable: bool
        """
        import sys
        import os
        # if console is not 'stdout' or 'stderr' raise ValueError
        assert console in ['stdout', 'stderr']
        self.console = sys.stdout if console == "stdout" else sys.stderr
        self.file = fpath # file path
        self.f = None # file
        self.immediately_visiable = immediately_visiable # that means the log will be written to the file immediately

        if fpath is not None:
            # Remove existing log file
            if os.path.exists(fpath):
                os.remove(fpath)
        if console == 'stdout':
            sys.stdout = self
        else:
            sys.stderr = self

    # close the file
    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    # close the file
    def __exit__(self, **args):
        self.close()

    # write the msg to the file
    def write(self, msg):
        """
        write the msg to the file
            Args:
                msg: message
                
            Returns:
                None
        """
        self.console.write(msg)
        if self.file is not None:
            # if the directory of the file does not exist, create it
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
                
    # flush the file
    def flush(self):
        """
        flush the file, means write the file to the disk
            Args:
                None
                
            Returns:
                None
        """
        self.console.flush()
        if self.f is not None:
            # flush the write buffers of the file
            self.f.flush()
            import os
            os.fsync(self.f.fileno())

    # close the file
    def close(self):
        """
        restore the sys.stdout or sys.stderr
            Args:
                None
            
            Returns:
                None
        """
        self.console.close()
        if self.f is not None:
            self.f.close()
