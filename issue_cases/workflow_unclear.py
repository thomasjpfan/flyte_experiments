from flytekit import task, workflow


@task
def add(a: int, b: int) -> int:
    return a + b


@workflow
def run(a: int, b: int) -> int:
    return add(b=b)
