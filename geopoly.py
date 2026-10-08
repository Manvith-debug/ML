"""One command-line entry point for experiments, prediction, checks, and reports."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True

PROJECT = Path(__file__).resolve().parent
# Optional local dependency directory used by this workspace; normal installs use pip.
if (PROJECT.parent / '.deps').exists():
    sys.path.insert(0, str(PROJECT.parent / '.deps'))

parser = argparse.ArgumentParser(description='BT2024261 polynomial modeling laboratory')
commands = parser.add_subparsers(dest='command', required=True)
train = commands.add_parser('train', help='Run the full configuration-driven experiment')
train.add_argument('--data', type=Path, default=PROJECT)
predict = commands.add_parser('predict', help='Predict with an existing model')
predict.add_argument('--model', type=Path, required=True)
predict.add_argument('--input', type=Path, required=True)
predict.add_argument('--output', type=Path, required=True)
check = commands.add_parser('check', help='Check data separation, models, and deliverables')
check.add_argument('--data', type=Path, default=PROJECT)
report = commands.add_parser('report', help='Create the experiment notebook PDF')
report.add_argument('--repository', default='https://github.com/Manvith-debug/ML')
args = parser.parse_args()
if args.command == 'train':
    from experiment import run_project
    run_project(PROJECT, args.data)
elif args.command == 'predict':
    from experiment import infer
    infer(args.model, args.input, args.output)
elif args.command == 'check':
    from checks import verify
    verify(PROJECT, args.data)
else:
    from notebook import build
    build(PROJECT, args.repository)

