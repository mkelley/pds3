# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""units
========

Define PDS3 units.

PDS3 allows unit names to be plural, so add those here.

Also add ``localday``.

Add these to your default astropy unit namespace with ``enable()``.

"""

from astropy.units.utils import generate_unit_summary as _generate_unit_summary
import astropy.units as u

_namespace = globals()

localday = u.def_unit('localday', namespace=_namespace)
localhour = u.def_unit('localday/24', localday/24, namespace=_namespace)

# From Allison & McEwen 2000, Planetary and Space Science 48, 215
# martian_sol = u.Equivalency([(localday,
#                               u.d,
#                               lambda x: x * 1.02749125,
#                               lambda x: x / 1.02749125)])

# units from PDS3 standards reference 3.7
meters = u.def_unit('meters', u.m, namespace=_namespace)
grams = u.def_unit('grams', u.g, namespace=_namespace)
seconds = u.def_unit('seconds', u.s, namespace=_namespace)
amperes = u.def_unit('amperes', u.A, namespace=_namespace)
kelvins = u.def_unit('kelvins', u.K, namespace=_namespace)
moles = u.def_unit('moles', u.mol, namespace=_namespace)
candelas = u.def_unit('candelas', u.cd, namespace=_namespace)
radians = u.def_unit('radians', u.rad, namespace=_namespace)
steradians = u.def_unit('steradians', u.sr, namespace=_namespace)

###########################################################################
# DOCSTRING

# This generates a docstring for this module that describes all of the
# standard units defined here.
if __doc__ is not None:
    __doc__ += _generate_unit_summary(globals())


def enable():
    """
    Enable PDS3 units so they appear in results of
    `~astropy.units.UnitBase.find_equivalent_units` and
    `~astropy.units.UnitBase.compose`.

    This may be used with the ``with`` statement to enable Imperial
    units only temporarily.

    """
    # Local import to avoid cyclical import
    from astropy.units.core import add_enabled_units
    # Local import to avoid polluting namespace
    import inspect
    return add_enabled_units(inspect.getmodule(enable))
