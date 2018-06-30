from time import time

import sequence_repeater
from genetic import Runner


class ExperimentRunner(Runner):
    @staticmethod
    def build_agent():
        env = sequence_repeater.Game(max_num=4, max_timesteps=100)
        return sequence_repeater.Agent(env=env)

    @classmethod
    def run_experiment(cls):
        from experiments.util import get_empty_data_file, get_or_make_data_dir

        saved_agents_dir = get_or_make_data_dir('agents')

        r = cls.__new__(cls)
        r.__init__(agent_builder=ExperimentRunner.build_agent, num_agents=200, num_champions=2,
                   info_file_path=get_empty_data_file('data.csv'), max_workers=8)

        generations = 10
        for s in range(1, generations + 1):
            start_time = time()
            # Run evaluation.
            f, info = r.evaluate()
            end_time = time()
            avg_fitness = sum(f) / len(f)
            print(f"Generation {s} \t"
                  f"Fitness: {avg_fitness}\t"
                  f"Best: {max(f)}\t"
                  f"in {end_time-start_time} s")
            # Record info to log.
            r.record_info()
            # Remove the old agents and save the current ones.
            r.save_agents(directory=saved_agents_dir, overwrite=True)

            # Breed next generation.
            r.breed()


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
