import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()

parser.add_argument('--var', help='var', type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument('--my_str', help='my_str', type=str)

args = parser.parse_args()

print(args.var)
print(args.my_str)
print(type(args.var))
