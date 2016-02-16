pds3
====

Python module to read NASA Planetary Data System v3 files or data.
These functions are rudimentary, and should not be used for
validation.

Requires: numpy, astropy, ply.

Time fields are assumed to be on the UTC scale.  I haven't verified if
that assumption is valid for PDS3.

ASCII tables can be read, and there is some support for 2D images.

Problems?  Please open an issue, or submit a pull request with updated code.

- Michael S. Kelley, UMD

Caution
=======

I hope you find pds3 useful, but use at your own risk.

pds3 does not validate files, may work for invalid files, and may not
work for valid files.

