from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')

class Validator(Generic[T], ABC):

    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None) -> T:
        return getattr(obj, self.private_name)

    def __set__(self, obj, value: T) -> None:
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value: T) -> None:
        pass



class PositiveInt(Validator[int]):
    def validate(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"{value} is not a positive integer.")
        

class CouldBeNone(Validator[T | None]):
    def __init__(self, validator: Validator[T]) -> None:
        self.validator = validator

    def validate(self, value: T | None) -> None:
        if value is not None:
            self.validator.validate(value)