import pygame

from snake.direction import Direction

KEY_TO_ACTION_MAP = {
    pygame.K_w: Direction.UP,
    pygame.K_a: Direction.LEFT,
    pygame.K_s: Direction.DOWN,
    pygame.K_d: Direction.RIGHT,
    pygame.K_UP: Direction.UP,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_RIGHT: Direction.RIGHT,
}


def play(env=None):
    pygame.init()
    env.reset()
    env.render(mode='human')

    def do_game_loop():
        update_clock = pygame.time.Clock()
        while True:
            # Render.
            env.render(mode='human')

            # Process events.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # User closed window
                    return
                elif event.type == pygame.KEYDOWN:
                    # Skip unbound keys.
                    if event.key not in KEY_TO_ACTION_MAP:
                        continue

                    # Act on bound keys.
                    observation, reward, done, info = env.step(KEY_TO_ACTION_MAP[event.key])
                    print(f"Observation: {observation}\tReward: {reward}\tDone: {done}\tInfo: {info}")
                    if done:
                        env.reset()

            # Limit frame rate.
            update_clock.tick(30)

    do_game_loop()

    pygame.quit()


if __name__ == "__main__":
    from snake import Game
    from snake.observation_strategies.default_observation_strategy import DefaultObservationStrategy
    from snake.observation_strategies.inverse_distance_observation_strategy import InverseDistanceObservationStrategy

    play(Game(map_size=[10, 10], initial_snake_length=3, observation_strategy=InverseDistanceObservationStrategy))
