"""
Testing script for furniture assembly tasks
"""

from env.furniture_baxter_assembly import FurnitureBaxterAssemblyEnv
# import high level planning
# import low level planning

test_furniture = ['chair_agne-0007',
				  'chair_bernhard_0146',
				  'desk_mikael_1064',
				  'shelf-ivar_0678',
				  'swivel_chair_0700',
				  'table_klubbo_0743',
				  'table_lack_0825']

test_backgrounds = ['Industrial', 'Garage', 'Interior']

num_trials = 3

"""
Run experiments based on testing conditions
"""
def run_experiments():
	# initialize data collection
	high_level_planning_time = {}
	low_level_planning_time = {}
	experiment_success = {}

	# run experiment for each (furniture, background) test condition
	for f in test_furniture:
		for b in test_backgrounds:
			for i in range(num_trials):
				# initialize environment
				env = FurnitureBaxterAssemblyEnv(f, b)
				env.set_random_initial_state()

				# high-level planning
				high_level_task_plan, high_level_plan_time = ... # TODO
				high_level_planning_time[(f, b)].append(high_level_plan_time)

				# low-level planning
				low_level_task_plan, low_level_plan_time = ... # TODO
				low_level_planning_time[(f, b)].append(low_level_plan_time)

				# execute task and record video
				res = env.run_task(low_level_task_plan)
				experiment_success[(f, b)].append(res)

	write_data_to_file(high_level_planning_time, low_level_planning_time, experiment_success)

"""
Writes data to file and computes experiment statistics.
"""
def write_data_to_file(high_level_times, low_level_times, experiment_success):
	# TODO implement
	raise NotImplementedError

if __name__ == "__main__":
	run_experiments()