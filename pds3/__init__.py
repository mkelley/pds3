# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
pds3 --- Simple (minded) PDS3 tools
===================================

Classes
-------
PDS3Label        - Dictionary-like class for PDS3 label content.

Functions
---------
read_label       - Read a PDS3 label.
read_ascii_table - Read an ASCII table as described by a label.
read_image       - Read an image as described by the label.
read_table       - Read a table as described by a label.

"""

__all__ = [
    'read_label',
    'read_ascii_table',
    'read_image',
    'read_table'
]

from .core import *
from .pds3label import *
from . import units