from abc import abstractmethod


class InfoEmitter:
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def emit(self, game: 'snake.game.Game'):
        pass


class PropertyEmitter(InfoEmitter):
    """Simply emits a property from the game given its name."""

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def emit(self, game: 'snake.game.Game'):
        return vars(game)[self._name]