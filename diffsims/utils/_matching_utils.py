# -*- coding: utf-8 -*-
# Copyright 2017-2020 The diffsims developers
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

""" Exported from pyxem these utils are used to match patterns and
templates within Fourier space """

def _optimal_fft_size(target, real=False):
    """Wrapper around scipy function next_fast_len() for calculating optimal FFT padding.

    scipy.fft was only added in 1.4.0, so we fall back to scipy.fftpack
    if it is not available. The main difference is that next_fast_len()
    does not take a second argument in the older implementation.

    Parameters
    ----------
    target : int
        Length to start searching from. Must be a positive integer.
    real : bool, optional
        True if the FFT involves real input or output, only available
        for scipy > 1.4.0

    Returns
    -------
    int
        Optimal FFT size.
    """

    try:  # pragma: no cover
        from scipy.fft import next_fast_len

        support_real = True

    except ImportError:  # pragma: no cover
        from scipy.fftpack import next_fast_len

        support_real = False

    if support_real:  # pragma: no cover
        return next_fast_len(target, real)
    else:  # pragma: no cover
        return next_fast_len(target)


def _get_fourier_transform(template_coordinates, template_intensities, shape, fsize):
    """Returns the Fourier transform of a list of templates.

    Takes a list of template coordinates and the corresponding list of
    template intensities, and returns the Fourier transform of the template.

    Parameters
    ----------
    template_coordinates: numpy array
        Array containing coordinates for non-zero intensities in the template
    template_intensities: list
        List of intensity values for the template.
    shape: tuple
        Dimensions of the signal.
    fsize: list
        Dimensions of the Fourier transformed signal.

    Returns
    -------
    template_FT: numpy array
        Fourier transform of the template.
    template_norm: float
        Self correlation value for the template.
    """
    template = np.zeros((shape))
    template[
        template_coordinates[:, 1], template_coordinates[:, 0]
    ] = template_intensities[:]
    template_FT = np.fft.fftshift(np.fft.rfftn(template, fsize))
    template_norm = np.sqrt(full_frame_correlation(template_FT, 1, template_FT, 1))
    return template_FT, template_norm


def _get_library_FT_dict(template_library, shape, fsize):
    """Takes a template library and converts it to a dictionary of Fourier transformed templates.

    Parameters
    ----------
    template_library: DiffractionLibrary
        The library of simulated diffraction patterns for indexation.
    shape: tuple
        Dimensions of the signal.
    fsize: list
        Dimensions of the Fourier transformed signal.

    Returns
    -------
    library_FT_dict: dict
        Dictionary containing the fourier transformed template library, together with the corresponding orientations and
        pattern norms.

    """
    library_FT_dict = {}
    for entry, library_entry in enumerate(template_library.values()):
        orientations = library_entry["orientations"]
        pixel_coords = library_entry["pixel_coords"]
        intensities = library_entry["intensities"]
        template_FTs = []
        pattern_norms = []
        for coord, intensity in zip(pixel_coords, intensities):
            template_FT, pattern_norm = get_fourier_transform(
                coord, intensity, shape, fsize
            )
            template_FTs.append(template_FT)
            pattern_norms.append(pattern_norm)

        library_FT_dict[entry] = {
            "orientations": orientations,
            "patterns": template_FTs,
            "pattern_norms": pattern_norms,
        }

    return library_FT_dict

def _full_frame_correlation(image_FT, image_norm, pattern_FT, pattern_norm):
    """Computes the correlation score between an image and a template in Fourier space.

    Parameters
    ----------
    image: numpy.ndarray
        Intensities of the image in fourier space, stored in a NxM numpy array
    image_norm: float
        The norm of the real space image, corresponding to image_FT
    fsize: numpy.ndarray
        The size of image_FT, for us in transform of template.
    template_coordinates: numpy array
        Array containing coordinates for non-zero intensities in the template
    template_intensities: list
        List of intensity values for the template.

    Returns
    -------
    corr_local: float
        Correlation score between image and template.

    See Also
    --------
    correlate_library, fast_correlation, zero_mean_normalized_correlation

    References
    ----------
    A. Foden, D. M. Collins, A. J. Wilkinson and T. B. Britton "Indexing electron backscatter diffraction patterns with
     a refined template matching approach" doi: https://doi.org/10.1016/j.ultramic.2019.112845
    """

    fprod = pattern_FT * image_FT

    res_matrix = np.fft.ifftn(fprod)
    fsize = res_matrix.shape
    corr_local = np.max(
        np.real(
            res_matrix[
                max(fsize[0] // 2 - 3, 0) : min(fsize[0] // 2 + 3, fsize[0]),
                max(fsize[1] // 2 - 3, 0) : min(fsize[1] // 2 + 3, fsize[1]),
            ]
        )
    )
    if image_norm > 0 and pattern_norm > 0:
        corr_local = corr_local / (image_norm * pattern_norm)

    # Sub-pixel refinement can be done here - Equation (5) in reference article

    return corr_local

def _test_get_fourier_transform():
    shape = (3, 3)
    fsize = (5, 5)
    normalization_constant = 0.9278426705718053  # Precomputed normalization. Formula full_frame(template, 1, template, 1)
    template_coordinates = np.asarray([[1, 1]])
    template_intensities = np.asarray([1])
    transform, norm = get_fourier_transform(
        template_coordinates, template_intensities, shape, fsize
    )
    test_value = np.real(transform[2, 1])  # Center value
    np.testing.assert_approx_equal(test_value, 1)
    np.testing.assert_approx_equal(norm, normalization_constant)


def _test_get_library_FT_dict():
    new_template_library = DiffractionLibrary()
    new_template_library["GaSb"] = {
        "orientations": np.array([[0.0, 0.0, 0.0],]),
        "pixel_coords": np.array([np.asarray([[1, 1],])]),
        "intensities": np.array([np.array([1,])]),
    }
    shape = (3, 3)
    fsize = (5, 5)
    normalization_constant = 0.9278426705718053
    new_template_dict = get_library_FT_dict(new_template_library, shape, fsize)
    for phase_index, library_entry in enumerate(new_template_dict.values()):
        orientations = library_entry["orientations"]
        patterns = library_entry["patterns"]
        pattern_norms = library_entry["pattern_norms"]
    np.testing.assert_approx_equal(orientations[0][0], 0.0)
    np.testing.assert_approx_equal(np.real(patterns[0][2, 1]), 1)
    np.testing.assert_approx_equal(pattern_norms[0], normalization_constant)

def _test_full_frame_correlation():
    # Define testing parameters.
    in1 = np.zeros((10, 10))
    in2 = np.zeros((10, 10))
    in1[5, 5] = 1
    in1[7, 7] = 1
    in1[3, 7] = 1
    in1[7, 3] = 1
    in1_FT = np.fft.fftshift(np.fft.rfftn(in1, (20, 20)))
    norm_1 = np.sqrt(np.max(np.real(np.fft.ifftn(in1_FT ** 2))))
    in2_FT = np.fft.fftshift(np.fft.rfftn(in2, (20, 20)))
    norm_2 = np.sqrt(np.max(np.real(np.fft.ifftn(in2_FT ** 2))))
    np.testing.assert_approx_equal(
        full_frame_correlation(in1_FT, norm_1, in1_FT, norm_1), 1
    )
    np.testing.assert_approx_equal(
        full_frame_correlation(in1_FT, norm_1, in2_FT, norm_2), 0
    )


def _test_optimal_fft_size():
    np.testing.assert_approx_equal(optimal_fft_size(8), 8)
    np.testing.assert_approx_equal(optimal_fft_size(20), 20)
