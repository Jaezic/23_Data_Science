from collections import OrderedDict
import os
import pprint
from config import argument_parser
from dataset.Dataset import FireDataset
from models.model import build_model
from tools.evaluate import evaluate
from tools.utils import ReDirectSTD, set_seed, time_str
from tools.visualization import visual


def main(args):
    set_seed(args.seed)  # Set random seed

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

    # Dataset setup
    dataset = FireDataset(args)

    # Model setup
    model = build_model(args)

    # Training setup
    trainer(args, model, dataset)

    # Evaluation setup
    validate(args, model, dataset)


def trainer(args, model, dataset):
    model.fit(dataset.x, dataset.y)


def validate(args, model, dataset):
    y = model.predict(dataset.x)

    visual(dataset, y)

    evaluate(model, dataset, y)


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
