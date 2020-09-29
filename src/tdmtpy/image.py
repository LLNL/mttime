# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-814839
# author: Andrea Chiang (andrea4@llnl.gov)
"""
Plotting routines for tdmtpy

.. warning::

   This module should NOT be used directly instead used the class method
   :meth:`~tdmtpy.inversion.Inversion.plot`.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath

from matplotlib.gridspec import GridSpec
import matplotlib.collections as mpl_collections
from matplotlib import patches, transforms
from obspy.imaging.beachball import xy2patch
from obspy.imaging.scripts.mopad import BeachBall as mopad_BeachBall
from obspy.imaging.scripts.mopad import MomentTensor as mopad_MomentTensor
from obspy.imaging.scripts.mopad import epsilon

from obspy.geodetics.base import kilometers2degrees
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


def new_page(nsta,nrows,ncols,annot="",offset=2,figsize=(8.5,11)):
    """
    Creates a new figure

    Creates a new :class:`~matplotlib.figure.Figure` and customizes
    subplot layout and figure axes. This is the figure layout for plotting
    focal mechanisms and waveform fits.

    :param nsta: total number of stations.
    :param nrows: number of stations to plot per page.
    :param ncols: number of components to plot per page.
    :param annot: figure annotations.
    :param offset: offset between waveform traces.
    :param figsize: figure dimension (width,height) in inches.
    :return: figure container, axis of the focal mechanism plot, and axes for the waveform traces.
    :rtype: class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`, list
    """
    gs = GridSpec(nrows+offset,ncols,hspace=0.5,wspace=0.15)
    f = plt.figure(figsize=figsize)
    
    # Annotations and beach balls
    ax0 = f.add_subplot(gs[0:offset,:],xlim=(-5,3.55),ylim=(-0.75,0.6),aspect="equal")
    ax0.text(-5,0,annot,fontsize=10,verticalalignment='center')
    ax0.set_axis_off()

    # Waveforms
    ax1 = np.empty((nsta,3),dtype=np.object) # create empty axes
    for i in range(nsta):
        ax1[i,0] = f.add_subplot(gs[i+offset,0])
        ax1[i,1] = f.add_subplot(gs[i+offset,1])
        ax1[i,2] = f.add_subplot(gs[i+offset,2])

    # Adjust axes
    for i in range(nsta-1):
        adjust_spines(ax1[i,0],['left'])
        adjust_spines(ax1[i,1], [])
        adjust_spines(ax1[i,2], ["bottom"])
    adjust_spines(ax1[-1,0],['left',"bottom"])
    adjust_spines(ax1[-1,1],['bottom'])
    adjust_spines(ax1[-1,2],['bottom'])
    
    # Title
    ax1[0,0].set_title('Vertical',verticalalignment='bottom',fontsize=10,pad=15)
    ax1[0,1].set_title('Radial',verticalalignment='bottom',fontsize=10,pad=15)
    ax1[0,2].set_title('Tangential',verticalalignment='bottom',fontsize=10,pad=15)
    
    return (f,ax0,ax1)


def adjust_spines(ax,spines):
    """
    Adjust axis spine

    Function to customize figure axes.

    :param ax: the Axes instance containing the spine.
    :type ax: :class:`~matplotlib.axes.Axes`
    :param spines: spine type.
    :type spines: list(str)
    """
    ax.tick_params(direction='in',labelsize=7)
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',5)) # outward by 5 points
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('time (s)')
    else:
        ax.xaxis.set_ticks([])


def beach(fm, linewidth=1, facecolor='0.75', bgcolor='w', edgecolor='k',
          alpha=1.0, xy=(0, 0), width=200, size=100, nofill=False,
          zorder=100, mopad_basis='NED', axes=None, show_iso=False):
    """
    Plot beach ball based on `MoPaD <http://www.larskrieger.de/mopad/>`_

    Function that returns a beach ball as a collection and can be added to
    an existing :class:`~matplotlib.axes.Axes`. This is modified from the
    :func:`~obspy.imaging.mopad_wrapper.beach` function to properly handle isotropic
    components. Original function only supports pure isotropic sources when isotropic
    components are plotted.

    :param fm: focal mechanism in (strike, dip, rake) or (M11,M22,M33,M12,M13,M23).
        The moment tensor elements are given for a coordinate system with axes pointing
        in three directions. Default is North, East and Down.
    :type fm: list
    :param linewidth: width of nodal and border lines. Default is ``1``.
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
    :type width: int
    :param size: number of points interpolated to draw the cruve. Default is ``100``.
    :type size: int
    :param nofill: no shading of the beach ball. Default is ``False``.
    :type nofill: bool
    :param zorder: set the zorder for the artist. Artists with lower zorder values are drawn first.
        Default is ``100``.
    :type zorder: float
    :param mopad_basis: moment tensor coordinate system. See supported coordinate
        system below. Default is ``"NED"``.
    :type mopad_basis: str
    :param axes: figure axis for beach ball, this is used to ensure the aspect ratio is
        adjusted so that the beach ball is circular on non-scaled axes.
        Default is ``None``.
    :type axes: :class:`~matplotlib.axes.Axes`
    :param show_iso: flag to display the isotropic component, default is ``False``.
    :type show_iso: bool
    :return: a collection of lines and polygons.
    :rtype: :class:`~matplotlib.collections.PatchCollection`

    .. rubric:: Supported coordinate systems:

    .. cssclass: table-striped

    ================   ==================   ==========================
    ``mopad_basis``    vectors              reference
    ================   ==================   ==========================
    "NED"              north, east, down    Jost and Herrmann, 1989
    "USE"              up, south, east      Larson et al., 2010
    "XYZ"              east, north, up      Jost and Herrmann, 1989
    "NWU"              north, west, up      Stein and Wysession, 2003
    ================   ==================   ==========================
    """
    # initialize beachball
    mt = mopad_MomentTensor(fm, system=mopad_basis)
    bb = mopad_BeachBall(mt, npoints=size)

    ## Snippets added by A. Chiang
    if show_iso:
        bb._plot_isotropic_part = True
        bb._nodallines_in_NED_system()
    
    ##
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
    # Use the given axes to maintain the aspect ratio of beac hballs on figure
    # resize.
    if axes is not None:
        # This is what holds the aspect ratio (but breaks the positioning)
        collection.set_transform(transforms.IdentityTransform()) # Need to find another way to print as vector graphics
        # Next is a dirty hack to fix the positioning:
        # 1. Need to bring the all patches to the origin (0, 0).
        for p in collection._paths:
            p.vertices -= xy
        # 2. Then use the offset property of the collection to position the
        # patches
        collection.set_offsets(xy)
        collection._transOffset = axes.transData
    collection.set_edgecolors(edgecolor)
    collection.set_alpha(alpha)
    collection.set_linewidth(linewidth)
    collection.set_zorder(zorder)

    return collection


def beach_mw_depth(tensors,depth,event,show,format):
    """
    Plot depth search result

    A function that plots focal mechanism, variance reduction, and moment magnitude
    as a function of source depth.

    :param tensors: moment tensor solutions at various depths.
    :type tensors: a list of :class:`~tdmtpy.tensor.Tensor`
    :param depth: depth of solution with maximum variance reduction.
    :type depth: float
    :param event: event origin time, longitude and latitude, refer to :class:`~tdmtpy.configure.Configure`
        for details.
    :type event: dict
    :param show: Turn on interactive display.
    :type show: bool
    :param format: figure file format.
    :type format: str
    """
    # Can only be saved as a raster image due to the beach ball display
    # need to find another way to fix the beach ball aspect ratios without using
    # display coordinates

    # Turn interactive plotting off
    plt.ioff()  # only display plots when called, save figure without displaying in ipython

    fig = plt.figure(dpi=300)
    fig.set_size_inches(8.5,5)
    if format != "png":
        ax1 = fig.add_subplot(1, 1, 1, rasterized=True)
    else:
        ax1 = fig.add_subplot(1, 1, 1,)

    # Axis 1 - variance reduction
    color1 = "red"
    ax1.set_xlabel("Depth [km]")
    ax1.set_ylabel("VR [%]", color=color1)
    ax1.set_ylim([0, 120])
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_yticklabels([0, 20, 40, 60, 80, 100])
    title = "%s | %5.2f %5.2f"%(event["datetime"],event["longitude"],event["latitude"])
    ax1.set_title(title)

    color2 = "mediumblue"
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel("Mw", color=color2)
    ax2.set_xlim([0, 40])
    ax2.set_ylim([0, 12])
    ax2.set_yticks([0, 2, 4, 6, 8, 10])
    ax2.set_yticklabels([0, 2, 4, 6, 8, 10])

    for tensor in tensors:
        ax1.vlines(depth, 0, 100, color="black")
        ax1.plot(tensor.inverted.depth,tensor.inverted.total_VR,"o",color=color1)
        bb = beach(tensor.m,
                   xy=(tensor.inverted.depth,tensor.inverted.total_VR+10),
                   facecolor=color1,
                   width=90,
                   show_iso=True,
                   axes=ax1,
                  )
        ax1.add_collection(bb)
        ax1.tick_params(axis="y", labelcolor=color1)

    ax2.plot(tensor.inverted.depth, tensor.mw, "o", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.tight_layout() # otherwise the right y-label is slightly clipped
    outfile = "depth.bbmw.%s"%format
    fig.savefig(outfile,format=format,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)