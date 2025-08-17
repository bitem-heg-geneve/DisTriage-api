
from typing import Iterable, List

def chunked(iterable: Iterable, n: int):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
