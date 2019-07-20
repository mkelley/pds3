#!/usr/bin/env python
from numpy.distutils.core import setup

if __name__ == "__main__":
    from glob import glob

    setup(name='pds3',
          version='0.1.0',
          description='Read NASA Planetary Data System v3 files.',
          author="Michael S. Kelley",
          author_email="msk@astro.umd.edu",
          url="https://github.com/mkelley/pds3",
          packages=['pds3'],
          requires=['numpy', 'ply', 'astropy'],
          license='BSD',
          classifiers = [
              'Intended Audience :: Science/Research',
              "License :: OSI Approved :: BSD License",
              'Operating System :: OS Independent',
              "Programming Language :: Python :: 3.5",
              'Topic :: Scientific/Engineering :: Astronomy'
          ]
      )
