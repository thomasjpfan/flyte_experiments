from flytekit import task


@task
def an_error():
    1 / 0
