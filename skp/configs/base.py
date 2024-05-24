from types import SimpleNamespace


class Config(SimpleNamespace):

    def __getattribute__(self, value):
        try:
            return super().__getattribute__(value)
        except AttributeError:
            return None  
