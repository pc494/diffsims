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

import numpy as np
from transforms3d.euler import axangle2euler

def get_rotation_from_z(structure,direction):
    """
    Finds the rotation that takes [001] to a given zone axis.

    Parameters
    ----------
    structure : diffpy.structure

    direction : array like
        [UVW]

    Returns
    -------
    euler_angles : tuple
        'rzxz' in degrees

    See Also
    --------
    generate_zap_map
    get_grid_around_beam_direction

    Notes
    -----
    This implementation works with an initial implementation that has +x as left to right,
    +y as bottom to top and +z as out of the plane of a page. Rotatins are counter clockwise
    as you look from the tip of the axis towards the origin
    """

    # Case where we don't need a rotation
    if np.dot(direction,[0,0,1]) == np.linalg.norm(direction):
        return (0,0,0)

    # Normalize our directions
    cartesian_direction = structure.lattice.cartesian(direction)
    cartesian_direction = cartesian_direction / np.linalg.norm(cartesian_direction)

    rotation_axis = np.cross([0,0,1],cartesian_direction)
    rotation_angle = np.arccos(np.dot([0,0,1],cartesian_direction))
    euler = axangle2euler(rotation_axis,rotation_angle,axes='rzxz')
    return np.rad2deg(euler)

def generate_directional_simulations(structure,simulator,direction_list,reciprocal_radius=1,**kwargs):
    """
    Produces simualtion of a structure aligned with certain axes

    Parameters
    ----------
    structure : diffpy.structure

    simulator :

    direction_list : list of lists
        A list of [UVW] indicies, eg) [[1,0,0],[1,1,0]]

    reciprocal_radius : float
        Default to 1

    Returns
    -------
    direction_dictionary : dict
        Keys are zone axes, values are simulations
    """

    direction_dictionary = {}
    for direction in direction_list:
        rotation_rzxz = get_rotation_from_z(structure,direction)
        simulation = simulator.calculate_ed_data(structure,reciprocal_radius,rotation=rotation_rzxz,**kwargs)
        direction_dictionary[direction] = simulation

    return direction_dictionary

def generate_zap_map(structure,simulator,system='cubic',reciprocal_radius=1,density='7',**kwargs):
    """
    Produces a number of zone axis patterns for a structure

    Parameters
    ----------
    structure : diffpy.structure

    simulator : diffsims.diffraction_generator

    system : str
        'cubic','hexagonal', 'trigonal', 'tetragonal','orthorhombic','monoclinic' or 'triclinic' - defaults to 'cubic'

    reciprocal_radius : float
        Default to 1

    density : str
        '3' for the corners or '7' (corners + 3 midpoints + 1 centroid)

    **kwargs :
        keyword arguments to be passed to simulator.calculate_ed_data()

    Returns
    -------
    zap_dictionary : dict
        Keys are zone axes, values are simulations

    Example
    -------
    #TODO: illustrate how to plot a bunch of sims
    """

    corners_dict = {'cubic': [(0, 0, 1), (1, 0, 1), (1, 1, 1)],
    'hexagonal': [(0, 0, 0, 1), (1, 0, -1, 0), (1, 1, -2, 0)],
    'orthorhombic': [(0, 0, 1), (1, 0, 0), (0, 1, 0)],
    'tetragonal': [(0, 0, 1), (1, 0, 0), (1, 1, 0)],
    'trigonal': [(0, 0, 0, 1), (0, -1, 1, 0), (1, -1, 0, 0)],
    'monoclinic': [(0, 0, 1), (0, 1, 0), (0, -1, 0)]}
    #TODO include triclinic

    if density == '3':
        direction_list = corners_dict[system]
    elif density == '7':
        pass #TODO: write a function to do this

    zap_dictionary = generate_directional_simulations(structure,simulator,direction_list,**kwargs)

    return zap_dictionary