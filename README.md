pds3 v0.2.1
==============

Python module to read NASA Planetary Data System v3 files or data.
These functions are rudimentary, and should not be used for
validation.

Requires: numpy, astropy, ply.

Time fields are assumed to be on the UTC scale.  I haven't verified if
that assumption is valid for PDS3.

ASCII tables can be read, and there is some support for 2D images.  Binary table reading is experimental (`pds3.core.read_binary_table`).

Problems?  Please open an issue, or submit a pull request with updated code.

- Michael S. P. Kelley, UMD

Usage
=====
Read your label into the **experimental** `PDS3Label` class and access
key values via indexing:

```python
lbl = PDS3Label('product.lbl')
r = lbl['RECORD_LENGTH']
shape = lbl['IMAGE']['LINES'], lbl['IMAGE']['LINE_SAMPLES']
```

Print the full label as a string:

```python
print(lbl)
```

I've tested round-tripping a short label from the International Halley
Watch archive and it worked, but more testing is needed.

Caution
=======

I hope you find pds3 useful, but use at your own risk.

pds3 does not validate files, may work for invalid files, and may not
work for valid files.

