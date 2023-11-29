import urllib
from flytekit import task, current_context, workflow


# @task
# def make_input() -> int:
#     return 10


@task
def wow(a: int) -> str:
    # context = current_context()
    # print("HELLLO")
    # print("execution_id:", context.execution_id)
    response = urllib.request.urlopen("http://172.20.10.4:8000")
    return response.read().decode("utf-8")


@workflow
def run_workflow() -> str:
    # a = make_input()
    return wow(a=10)
