from dataclasses import dataclass


@dataclass(init=False)
class Datapoint:
    id: str
    head: str
    prompt: str
    code: str