from gym import RewardWrapper


class CumulativeRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.accumulated_reward = 0

    def reset(self):
        self.accumulated_reward = 0
        return self.env.reset()

    def reward(self, reward):
        self.accumulated_reward += reward
        return self.accumulated_reward


if __name__ == "__main__":
    from snake import Game
    from snake.play_human import play

    env = Game(map_size=[10, 10])
    env = CumulativeRewardWrapper(env)
    play(env=env)
