# -*- coding: utf-8 -*-
# Copyright 2017-2019 The diffsims developers
#
# This file is part of diffsims.
#
# diffsims is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# diffsims is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with diffsims.  If not, see <http://www.gnu.org/licenses/>.

import pytest

import diffpy.structure
import numpy as np
from diffsims.generators.zap_map_generator import get_rotation_from_z

def test_zero_rotation_cases(default_structure):
    r_test = get_rotation_from_z(default_structure,[0,0,2])
    assert r_test == (0,0,0)

class TestOrthonormals():

    @pytest.fixture(params=[(3,3,3),(3,3,4),(3,4,5)])
    def sample_system(self,request):
        """Orthonormal structures"""
        a,b,c = request.param[0],request.param[1],request.param[2]
        latt = diffpy.structure.lattice.Lattice(a,b,c,90,90,90)
        atom = diffpy.structure.atom.Atom(atype='Ni',xyz=[0,0,0],lattice=latt)
        return diffpy.structure.Structure(atoms=[atom],lattice=latt)

    def test_rotation_to__static_x_axis(self,sample_system):
        r_to_x = get_rotation_from_z(sample_system,[1,0,0])
        assert np.allclose(r_to_x,(90,90,-90))

    def test_rotation_to_static_y_axis(self,sample_system):
        r_to_y = get_rotation_from_z(sample_system,[0,1,0])
        assert np.allclose(r_to_y,(180,90,-180))

    def test_rotations_to_static_yz(self,sample_system):
        """ We rotate from z towards y, and compare the results to geometry"""
        r_to_yz = get_rotation_from_z(sample_system,[0,1,1])
        tan_angle = np.tan(np.deg2rad(r_to_yz[1]))
        tan_lattice = sample_system.lattice.b / sample_system.lattice.c
        assert np.allclose(tan_angle,tan_lattice,atol=1e-5)


class TestHexagonal:
    """ Results are taken from """

    def test_rotation_to_streographic_corner_a(self):
        pass
    def test_rotation_to_streographic_corner_b(self):
        pass