import pytest
import astropy.units as u
from .. import units


def test_localday():
    with units.enable():
        assert str(1 * u.Unit('localday')) == '1.0 localday'


def test_localhour():
    with units.enable():
        assert u.isclose(1 * u.Unit('localday/24'), u.Unit('localday') / 24)


# def test_martian_sol():
#     with units.enable():
#         assert u.isclose((1 * u.Unit('localday')).to('s', units.martian_sol),
#                          88775.244 * u.s)


@pytest.mark.parametrize('unit,other', [
    ('meters', u.m),
    ('grams', u.g),
    ('amperes', u.A),
    ('kelvins', u.K),
    ('moles', u.mol),
    ('candelas', u.cd),
    ('radians', u.rad),
    ('steradians', u.sr),
])
def test_plural_unit(unit, other):
    with units.enable():
        assert u.Unit(unit).is_equivalent(other)
