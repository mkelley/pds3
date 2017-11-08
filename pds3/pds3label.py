# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""pds3label
============

Classes
-------
PDS3Label

"""

__all__ = ['PDS3Label']

from collections import OrderedDict

class PDS3Label(OrderedDict):
    """Dictionary-like class for PDS3 label content.

    Access keys via object indexing:

      lbl = PDS3Label('product.lbl')
      r = lbl['RECORD_LENGTH']
      shape = lbl['IMAGE']['LINES'], lbl['IMAGE']['LINE_SAMPLES']

    Parameters
    ----------
    fn : string, optional
      The name of the file to read.

    Attributes
    ----------
    raw : string
      The full label as a string, generated from the current (key,
      value) pairs.

    """
    
    def __init__(self, fn=None, raw=None):
        from .core import read_label
        self.update(read_label(fn))

    def __str__(self):
        return self._format_dict(self)

    def _format_dict(self, d, indent=0):
        from .core import PDS3Object, PDS3Group

        s = ""
        n = max([10] + [len(k) for k in d.keys()])
        format_key = lambda k: ('{:' + str(n) + '}').format(k)

        for k, v in d.items():
            if isinstance(v, (PDS3Object, PDS3Group)):
                if isinstance(v, PDS3Object):
                    name = 'OBJECT'
                else:
                    name = 'GROUP'
                    
                if not isinstance(v, list):
                    v = [v]
                    
                for i in range(len(v)):
                    s += '{}{} = {}\r\n'.format(
                        ' ' * indent, format_key(name), k)
                    s += self._format_dict(v[i], indent=indent + 2)
                    s += '{}{} = {}\r\n'.format(
                        ' ' * indent, format_key('END_' + name), k)
            else:
                s += '{}{} = {}\r\n'.format(
                    ' ' * indent, format_key(k), self._format_value(v))

        return s

    def _format_value(self, v):
        from astropy.time import Time
        from astropy.units import Quantity
        from .core import PDS3Object, PDS3Group, PDS3Keyword

        if isinstance(v, (float, int, PDS3Keyword)):
            s = str(v)
        elif isinstance(v, str):
            s = '"{}"'.format(v)
        elif isinstance(v, Time):
            s = v.isot
        elif isinstance(v, Quantity):
            s = '{} <{}>'.format(str(v.value), str(v.unit).upper())
        elif isinstance(v, set):
            s = '{{{}}}'.format(','.join([self._format_value(x) for x in v]))
        elif isinstance(v, (tuple, list)):
            s = '({})'.format(','.join([self._format_value(x) for x in v]))
        else:
            raise ValueError("Unknown value type: {} ({})".format(v, type(v)))

        return s
                
    @property
    def raw(self):
        s = self._format_dict(self) + 'END'
        rbytes = self['RECORD_BYTES']
        r = len(s) % rbytes
        if r:
            s += ' ' * (rbytes - r)
        return s
