"""
This module should only contain types. It is named "common.py" instead of
"types.py" to avoid shadowing the core python library named types.
"""
from typing import NewType

Epoch = NewType("Epoch", int)
DirPath = NewType("DirPath", str)
FilePath = NewType("FilePath", str)
