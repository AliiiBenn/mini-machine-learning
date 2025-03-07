from typing import TypeVar, overload
from collections.abc import Sequence


T = TypeVar('T')


class Labels(Sequence[T]):
    def __init__(self, *labels: T) -> None:
        self._labels = list(labels)

    def __len__(self) -> int:
        return len(self._labels)
    
    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]:
        ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        return self._labels[index]
    
    def __str__(self) -> str:
        return str(self._labels)
    
    def __repr__(self) -> str:
        return f"Labels({self._labels})"
    

    @property
    def classes(self) -> Sequence[T]:
        return list(set(self._labels))
    
    @property
    def classes_prior(self) -> dict[T, float]:
        return {c: self._labels.count(c) / len(self._labels) for c in self.classes}
