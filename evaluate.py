from collections import OrderedDict
import os
import pprint

from config import argument_parser
from tools.utils import ReDirectSTD, time_str
from main import main

if __name__ == '__main__':
    # Argument parsing
    parser = argument_parser()
    args = parser.parse_args()
    
    # Logging setup
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    if args.redirector:
        print('ReDirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)
    pprint.pprint(OrderedDict(args.__dict__))
    print('-' * 60)
    
    # Main for Training, Evaluation
    acc, pre, rec, f1 = main(args)
    
    