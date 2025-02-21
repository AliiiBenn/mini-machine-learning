from typing import Optional, Self, TypeAlias, TypeVar, Generic

from src.core.validators import CouldBeNone, PositiveInt


T = TypeVar("T")

class Node(Generic[T]):
    def __init__(self, 
                 value: T,
                 left: Optional[Self] = None,
                 right: Optional[Self] = None) -> None:
        self.value = value 
        self.left = left
        self.right = right 

    @property
    def is_leaf(self) -> bool:
        return self.value is None
    

class DecisionTreeNode(Node[float]):
    def __init__(self,
                 value: float,
                 left: Optional[Self] = None,
                 right: Optional[Self] = None,
                 feature_index: Optional[int] = None,
                 threshold: Optional[float] = None,
                 ) -> None:
        
        super().__init__(value, left, right)
        self.feature_index = feature_index
        self.threshold = threshold