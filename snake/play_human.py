import pygame

from snake.direction import Direction


def play(env=None):
    pygame.init()

    env.render(mode='human')
    KEY_TO_ACTION_MAP = {
        pygame.K_w: Direction.UP,
        pygame.K_a: Direction.LEFT,
        pygame.K_s: Direction.DOWN,
        pygame.K_d: Direction.RIGHT
    }

    def do_game_loop():
        update_clock = pygame.time.Clock()
        while True:
            # Render.
            env.render(mode='human')

            # Process events.
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    return
                elif event.type == pygame.KEYDOWN:
                    observation, reward, done, info = env.step(KEY_TO_ACTION_MAP[event.key])
                    print(f"Reward: {reward}\tDone: {done}\tInfo: {info}")
                    if done:
                        env.reset()

            # Limit frame rate.
            update_clock.tick(30)

    do_game_loop()

    pygame.quit()


if __name__ == "__main__":
    from snake import Game

    play(Game(map_size=[10, 10]))