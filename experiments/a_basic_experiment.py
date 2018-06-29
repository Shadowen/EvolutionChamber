from time import time

import numpy as np

import snake
from experiments.util import *
from genetic import Runner
from gym_util.forwarding_wrappers import ForwardingRewardWrapper
from gym_util import RecordRewardWrapper
from snake import Game, DistanceObservationGame


class ExperimentRunner(Runner):

    @staticmethod
    def build_agent():
        game = DistanceObservationGame(map_size=(80, 40), initial_snake_length=3)
        game = FitnessWrapper(game)
        game = RecordRewardWrapper(game)
        return snake.Agent(env=game, hidden_nodes=[18, 18])

    @classmethod
    def run_experiment(cls):
        np.random.seed(1)
        saved_agents_dir = get_or_make_data_dir('agents')
        info_path = get_empty_data_file('data.csv')

        r = cls.__new__(cls)
        r.__init__(agent_builder=cls.build_agent, num_agents=2000, num_champions=20, max_workers=8,
                   info_file_path=info_path)
        generations = 100

        human_display = False
        if human_display:
            import copy
            import threading
            from genetic import Agent, Genome

            human_display_agent: Agent = cls.build_agent()
            best_genome: Genome = human_display_agent.genome
            best_genome_lock: threading.Lock = threading.Lock()

            def do_human_display():
                # Rendering...
                while human_display_agent.genome is not None:
                    with best_genome_lock:
                        human_display_agent.genome = best_genome
                    human_display_agent.run_iteration(render=True)

            human_display_thread = threading.Thread(target=do_human_display)
            human_display_thread.start()

        for s in range(1, generations + 1):
            start_time = time()
            f, info = r.evaluate()
            end_time = time()
            avg_fitness = sum(f) / len(f)
            print(f"Generation {s} \t"
                  f"Fitness: {avg_fitness})\t"
                  f"Best: {max(f)}\t"
                  f"in {end_time-start_time} s")
            if human_display:
                with best_genome_lock:
                    best_genome = copy.deepcopy(r.agents[np.argmax(f)].genome)

            # Record info to log.
            r.record_info()
            # Remove the old agents and save the current ones.
            r.save_agents(directory=saved_agents_dir, overwrite=True)

            # Breed next generation.
            r.breed_next_generation()

        # Wait for threads to terminate.
        if human_display:
            human_display_thread.join()


class FitnessWrapper(ForwardingRewardWrapper):
    def __init__(self, env: Game):
        super().__init__(env)

    def reward(self, reward=0):
        snake_length = len(self.env.snake_tail) + 1
        if snake_length < 10:
            return (self.env.timesteps ** 2) * (2 ** snake_length)
        else:
            return (self.env.timesteps ** 2) * (2 ** 10) * (snake_length - 9)


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
