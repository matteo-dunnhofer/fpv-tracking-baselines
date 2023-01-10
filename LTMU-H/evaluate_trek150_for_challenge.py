import sys
sys.path.append('./TREK-150-toolkit')

from ltmuh import LTMUH
from toolkit.experiments import ExperimentTREK150

tracker = LTMUH()

root_dir = './' # set the path to TREK-150's root folder
exp = ExperimentTREK150(root_dir, result_dir='./', report_dir='./')

# Run experiment with the OPE, MSE, and HOI, protocols and save results
exp.run(tracker, protocol='ope', visualize=False)
exp.run(tracker, protocol='mse', visualize=False)
exp.run(tracker, protocol='hoi', visualize=False)

# If needed, generate a report for the protocol of interest
exp.report([tracker.name], protocol='ope')
exp.report([tracker.name], protocol='mse')
exp.report([tracker.name], protocol='hoi')
