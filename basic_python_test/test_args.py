import argparse
import sys

parser = argparse.ArgumentParser(description='help demo')
# 位置型参数 不需要带-就可以使用，但是这些参数是必须的
parser.add_argument('data',
                    help='path to dataset',metavar="DIR")
args = parser.parse_args()
