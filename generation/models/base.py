# generation/models/base.py

from abc import ABC, abstractmethod


class BaseGenerator(ABC):

    @abstractmethod
    def generate(
            self,
            prompt: str,
            batch_size: int,
            seed: int,
    ):
        pass