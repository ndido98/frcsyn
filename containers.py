from collections.abc import Sequence, Iterable
import torch


class StringArray(Sequence[str]):
    def __init__(self, elems: Iterable[str]) -> None:
        encoded = [elem.encode("utf-8") for elem in elems]
        lengths = [len(elem) for elem in encoded]
        self.storage = torch.tensor(bytearray().join(encoded), dtype=torch.uint8)
        self.offsets = torch.tensor([0] + lengths).cumsum(dim=0)
        self.n_elems = len(elems)

    def __len__(self) -> int:
        return self.n_elems
    
    def __getitem__(self, index: int) -> str:
        return bytes(self.storage[self.offsets[index]:self.offsets[index + 1]]).decode("utf-8")


class SupervisedArray(Sequence[tuple[str, int]]):
    def __init__(self, elems: Iterable[tuple[str, int]]) -> None:
        self.str_container = StringArray([elem[0] for elem in elems])
        self.int_container = torch.tensor([elem[1] for elem in elems], dtype=torch.int32)
        self.n_elems = len(elems)

    def __len__(self) -> int:
        return self.n_elems
    
    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.str_container[index], self.int_container[index].item()


class SupervisedCoupleArray(Sequence[tuple[tuple[str, str], int]]):
    def __init__(self, elems: Iterable[tuple[tuple[str, str], int]]) -> None:
        self.str_container1 = StringArray([elem[0][0] for elem in elems])
        self.str_container2 = StringArray([elem[0][1] for elem in elems])
        self.int_container = torch.tensor([elem[1] for elem in elems], dtype=torch.int32)
        self.n_elems = len(elems)

    def __len__(self) -> int:
        return self.n_elems
    
    def __getitem__(self, index: int) -> tuple[str, str, int]:
        return (self.str_container1[index], self.str_container2[index]), self.int_container[index].item()