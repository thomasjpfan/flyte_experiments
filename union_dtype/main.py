from typing import Union
from flytekit import task


@task
def hello_old_union(name: Union[str, int]) -> float:
    return 1.2


@task
def hello_new_union(name: str | int) -> float:
    return 1.2
