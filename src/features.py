from typing import TypeVar, overload    
from collections.abc import Sequence


T = TypeVar('T')


class Features(Sequence[Sequence[T]]):
    def __init__(self, *features: Sequence[T]) -> None:
        self._features = list(features)

    def __len__(self) -> int:
        return len(self._features)
    
    @overload
    def __getitem__(self, index: int) -> Sequence[T]:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Sequence[T]]:
        ...

    def __getitem__(self, index: int | slice) -> Sequence[T] | Sequence[Sequence[T]]:
        return self._features[index]
    
    def __str__(self) -> str:
        return str(self._features)
    
    def __repr__(self) -> str:
        return f"Features({self._features})"

