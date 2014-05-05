pds3
====

Python module to read NASA Planetary Data System v3 files or data.
These functions are rudimentary, and should not be used for
validation.

Requires: numpy, astropy, ply.

Time fields are assumed to be on the UTC scale.  I haven't verified if
that assumption is valid for PDS3.

Currently, only ASCII tables can be read, although I would like to
support binary tables.


Problems?  Please, send me the label or the code to fix it.

- Michael S. Kelley, UMD

Caution
=======

I hope you find pds3 useful, but use at your own risk.

pds3 does not validate files, may work for invalid files, and may not
work for valid files.

