import yaml
import argparse

import envs
import boostup
from boostup.utils.experiment import *

def load_yaml(filename):
    with open(filename, 'r') as stream:
        return yaml.safe_load(stream)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='configuration file')
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    
    learner = eval('boostup.'+config['algo'])
    exp_fn = eval(config['exp_type'])

    exp = exp_fn(learner, config)
    print('Starting the experiment run...')
    exp.run_experiment()