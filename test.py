from config import argument_parser
from dataset.Dataset import FireDataset

def main(args):
    dataset = FireDataset(args)
    print(dataset.x)
    print(dataset.y)
    print(len(dataset.x), len(dataset.y))

if __name__  == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)