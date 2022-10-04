import sys
sys.path.append('./TREK-150-toolkit')

from tbydh import TbyDH
from toolkit.experiments import ExperimentTREK150

tracker = TbyDH()

root_dir = './' # set the path to TREK-150's root folder
exp = ExperimentTREK150(root_dir, result_dir='./', report_dir='./')
prot = 'ope'

# Run an experiment with the protocol of interest and save results
exp.run(tracker, protocol=prot, visualize=False)

# Generate a report for the protocol of interest
exp.report([tracker.name], protocol=prot)