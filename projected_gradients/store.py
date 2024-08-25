from abc import ABC


class Store(dict, ABC):
    def __init__(self, names_of_params: list[str]):
        """
        Makes a store object with ndim vectors for the given parameter set
        """
        self.names_of_params = names_of_params
        self.store = {}

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def get_store(self):
        return self.store
