# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-814839
# author: Andrea Chiang (andrea@llnl.gov)
"""
Draws a beach ball diagram of a moment tensor

Source codes are adapted from the :class:`~obspy.imaging.mopad_wrapper`.

"""

import numpy as np
import matplotlib.collections as mpl_collections
from matplotlib import patches, transforms

from obspy.imaging.beachball import xy2patch
from .scripts.mopad import BeachBall
from .scripts.mopad import MomentTensor
from .scripts.mopad import epsilon


def beach(fm, linewidth=1, facecolor='0.75', bgcolor='w', edgecolor='k',
          alpha=1.0, xy=(0, 0), width=200, size=100, nofill=False,
          zorder=100, mopad_basis='NED', axes=None, show_iso=False):
    """
    Plot beach ball based on `MoPaD <http://www.larskrieger.de/mopad/>`_

    Function that returns a beach ball as a collection and can be added to
    an existing :class:`~matplotlib.axes.Axes`.

    :param fm: focal mechanism in (strike, dip, rake) or (M11,M22,M33,M12,M13,M23).
        The moment tensor elements are given for a coordinate system with axes pointing
        in three directions.
    :type fm: list
    :param linewidth: width of nodal and border lines. Default is ``0.8``.
    :type linewidth: float
    :param facecolor: color or shade of the compressive quadrants.
        Default is ``0.75`` (gray).
    :type facecolor: str
    :param bgcolor: background color, default is ``w`` (white).
    :type bgcolor: str
    :param edgecolor: color of nodal and border lines, default is ``b`` (black).
    :type edgecolor: str
    :param alpha: beach ball transparency, default is ``1.0`` (opaque).
    :type alpha: float
    :param xy: original position of the beach ball. Default is ``(0, 0)``
    :type xy: tuple
    :param width: width of the beach ball (aka symbol size). Default is ``200``.
    :type width: float
    :param size: number of points interpolated to draw the curve. Default is ``100``.
    :type size: int
    :param nofill: no shading of the beach ball. Default is ``False``.
    :type nofill: bool
    :param zorder: set the zorder for the artist. Artists with lower zorder values are drawn first.
        Default is ``100``.
    :type zorder: float
    :param mopad_basis: moment tensor coordinate system. Default is ``"NED"``.
    :type mopad_basis: str
    :param axes: figure axis for beach ball, this is used to ensure the aspect ratio is
        adjusted so that the beach ball is circular on non-scaled axes. When this option is used
        figure cannot be saved in vector format.
        Default is ``None``.
    :type axes: :class:`~matplotlib.axes.Axes`
    :param show_iso: flag to display the isotropic component, default is ``False``.
    :type show_iso: bool
    :return: a collection of lines and polygons.
    :rtype: :class:`~matplotlib.collections.PatchCollection`

    .. warning::

        Set ``axes=None`` if you want to save the beach ball image in vector file formats.

    """
    # initialize beachball
    mt = MomentTensor(fm, system=mopad_basis)
    bb = BeachBall(mt, npoints=size)

    if show_iso:
        # Include the isotropic component
        bb._plot_isotropic_part = True
        bb._nodallines_in_NED_system()
    bb._setup_BB(unit_circle=False)

    # extract the coordinates and colors of the lines
    radius = width / 2.0
    neg_nodalline = bb._nodalline_negative_final_US
    pos_nodalline = bb._nodalline_positive_final_US
    tension_colour = facecolor
    pressure_colour = bgcolor

    if nofill:
        tension_colour = 'none'
        pressure_colour = 'none'

    # based on mopads _setup_plot_US() function
    # collect patches for the selection
    coll = [None, None, None]
    coll[0] = patches.Circle(xy, radius=radius)
    coll[1] = xy2patch(neg_nodalline[0, :], neg_nodalline[1, :], radius, xy)
    coll[2] = xy2patch(pos_nodalline[0, :], pos_nodalline[1, :], radius, xy)

    # set the color of the three parts
    fc = [None, None, None]
    if bb._plot_clr_order > 0:
        fc[0] = pressure_colour
        fc[1] = tension_colour
        fc[2] = tension_colour
        if bb._plot_curve_in_curve != 0:
            fc[0] = tension_colour
            if bb._plot_curve_in_curve < 1:
                fc[1] = pressure_colour
                fc[2] = tension_colour
            else:
                coll = [coll[i] for i in (0, 2, 1)]
                fc[1] = pressure_colour
                fc[2] = tension_colour
    else:
        fc[0] = tension_colour
        fc[1] = pressure_colour
        fc[2] = pressure_colour
        if bb._plot_curve_in_curve != 0:
            fc[0] = pressure_colour
            if bb._plot_curve_in_curve < 1:
                fc[1] = tension_colour
                fc[2] = pressure_colour
            else:
                coll = [coll[i] for i in (0, 2, 1)]
                fc[1] = tension_colour
                fc[2] = pressure_colour

    if bb._pure_isotropic:
        if abs(np.trace(bb._M)) > epsilon:
            # use the circle as the most upper layer
            coll = [coll[0]]
            if bb._plot_clr_order < 0:
                fc = [tension_colour]
            else:
                fc = [pressure_colour]

    # transform the patches to a path collection and set
    # the appropriate attributes
    collection = mpl_collections.PatchCollection(coll, match_original=False)
    collection.set_facecolors(fc)
    # Use the given axes to maintain the aspect ratio of beachballs on figure
    # resize.
    if axes is not None:
        # This is what holds the aspect ratio (but breaks the positioning), and
        # the part that breaks when saving the image to a vector file format
        collection.set_transform(transforms.IdentityTransform())
        # Bring the all patches to the origin (0, 0).
        for p in collection._paths:
            p.vertices -= xy
        # Use the offset property of the collection to position the patches
        collection.set_offsets(xy)
        collection._transOffset = axes.transData
    collection.set_edgecolors(edgecolor)
    collection.set_alpha(alpha)
    collection.set_linewidth(linewidth)
    collection.set_zorder(zorder)

    return collection
