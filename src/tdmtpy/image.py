# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-814839
# author: Andrea Chiang (andrea4@llnl.gov)
"""
Plotting routines for tdmtpy

.. warning::

   This module should NOT be used directly instead used the method
   :meth:`~tdmtpy.inversion.Inversion.plot` in :class:`~tdmtpy.inversion.Inversion`
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


def beach_map(m,event,stlo,stla,distance,show,format,zoom_level=8):
    """
    Plot beach ball on a map

    Function to plot stations and focal mechanisms on a Stamen terrain background.

    :param m: focal mechanisms, refer to :func:`~tdmtpy.image.beach` function
        for the supported formats.
    :type m: list
    :param event: event origin time, longitude and latitude, refer to :class:`~tdmtpy.configure.Configure`
        for details.
    :type event: dict
    :param stlo: station longitudes
    :type stlo: list or :class:`~numpy.ndarray`
    :param stla: station latitudes.
    :type stla: list or :class:`~numpy.ndarray`
    :param distance: source-receiver distance.
    :type distance: list or :class:`~numpy.ndarray`
    :param show: Turn on interactive display.
    :type show: bool
    :param format: figure file format.
    :type format: str
    :param zoom_level: background image tile zoom level. Default is ``8``.
    :type zoom_level: int
    """
    # Turn interactive plotting off
    plt.ioff()  # only display plots when called, save figure without displaying in ipython

    # Calculate image extent based on epicentral distance
    width = kilometers2degrees(0.5 * max(distance))
    height = kilometers2degrees(0.5 * max(distance))

    lat1 = min(stla) - height
    lat2 = max(stla) + height
    lon1 = min(stlo) - width
    lon2 = max(stlo) + width

    data_crs = ccrs.PlateCarree()
    evlo = event["longitude"]
    evla = event["latitude"]
    point = (evlo,evla)

    stamen_terrain = cimgt.Stamen('terrain-background')
    projection = stamen_terrain.crs

    fig = plt.figure(dpi=300)
    fig.set_size_inches(8.5,11)
    if format != "png":
        ax = fig.add_subplot(1, 1, 1, projection=projection,rasterized=True)  # axes coordinates
    else:
        ax = fig.add_subplot(1, 1, 1, projection=projection)  # axes coordinates
    ax.set_extent([lon1, lon2, lat1, lat2])
    ax.add_image(stamen_terrain, zoom_level)

    # Add tick labels
    g1 = ax.gridlines(crs=data_crs,draw_labels=True)
    g1.top_labels = False

    # Plot stations
    ax.plot(stlo,stla,marker="^", color="black", markersize=8, linestyle="", transform=data_crs)

    # Plot beach ball on map
    x, y = projection.transform_point(*point, src_crs=data_crs)
    bb = beach(m,xy=(x,y),facecolor="red",width=135,show_iso=True,axes=ax)
    ax.add_collection(bb)

    # Add title
    ax.set_title(event["datetime"])

    outfile = "map.%s"%format
    fig.savefig(outfile,format=format,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def lune(gamma,delta,show,format):
    """
    Plot source-type on a lune

    Function to plot moment tensor source type on a lune based on the formulation
    of Tape and Tape (2012). Theoretical source types and source type arcs are also
    plotted.

    :param gamma: lune longitude.
    :type gamma: float
    :param delta: lune latitude.
    :type delta: float
    :param show: Turn on interactive display.
    :type show: bool
    :param format: figure file format.
    :type format: str
    """
    # Turn interactive plotting off
    plt.ioff()  # only display plots when called, save figure without displaying in ipython

    fig = plt.figure(figsize=[6,10])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertAzimuthalEqualArea())

    # Draw boundary around the lune
    longitude = np.concatenate([[-30], np.tile(-30, 180), [-30], np.tile(30, 180), [30]])
    latitude = np.concatenate([[-90], np.arange(-90, 90, 1), [90], np.arange(90, -90, -1), [-90]])
    codes = np.hstack([mpath.Path.MOVETO, [mpath.Path.LINETO] * 180,
                       mpath.Path.MOVETO, [mpath.Path.LINETO] * 180,
                       mpath.Path.MOVETO])
    verts = np.column_stack([longitude[::-1], latitude[::-1]])
    path = mpath.Path(verts, codes[::-1])
    ax.set_boundary(path, transform=ccrs.PlateCarree())
    ax.set_extent([-30, 30, -90, 90], ccrs.PlateCarree())

    # Add gridlines
    xlocs = np.arange(-30, 30 + 10, 10, dtype=np.int)
    ylocs = np.arange(-90, 90 + 10, 10, dtype=np.int)
    ax.gridlines(xlocs=xlocs, ylocs=ylocs)

    # Offset text by pixels
    text_transform = ccrs.PlateCarree()._as_mpl_transform(ax)
    pixels = 10
    left = transforms.offset_copy(text_transform, units="dots", x=-1 * pixels)
    right = transforms.offset_copy(text_transform, units="dots", x=pixels)
    top = transforms.offset_copy(text_transform, units="dots", y=pixels)
    bottom = transforms.offset_copy(text_transform, units="dots", y=-1 * pixels)

    # Theoretical sources
    x = [0, -30, -30, 0, 30, 30, -30, 30, 0]
    y = [-90, 0, 35.2644, 90, 0, -35.2644, 60.5038, -60.5038, 0]
    sources = ["-ISO", "+CLVD", "+LVD", "+ISO", "-CLVD", "-LVD", "+Crack", "-Crack", "DC"]
    offset = [bottom, left, left, top, right, right, left, right, bottom]

    halign = ["center", "right", "right", "center", "left", "left", "right", "left", "center"]
    valign = ["top", "center", "center", "bottom", "center", "center", "center", "center", "top"]
    ax.plot(x, y, "o", color="black", markersize=8, transform=ccrs.PlateCarree(), clip_on=False)
    for i in range(len(x)):
        ax.text(x[i], y[i], sources[i],
                transform=offset[i], clip_on=False,
                fontsize=14,
                verticalalignment=valign[i], horizontalalignment=halign[i])

    # Source type arc
    x = [-30.00,-29.20,-28.38,-27.55,-26.71,-25.86,-24.98,-24.10,-23.19,
         -22.27,-21.34,-20.39,-19.42,-18.43,-17.42,-16.40,-15.35,-14.29,
         -13.21,-12.10,-10.98, -9.83, -8.67, -7.48, -6.27, -5.04, -3.79,
          -2.51, -1.22,  0.10,  1.44,  2.79,  4.17,  5.57,  6.99,  8.43,
           9.89, 11.36, 12.85, 14.36, 15.88, 17.41, 18.96, 20.51, 22.08,
          23.65, 25.24, 26.82, 28.41, 30.00
         ]
    y = [ 35.26, 35.91, 36.55, 37.19, 37.82, 38.44, 39.06, 39.67, 40.27,
          40.87, 41.46, 42.04, 42.62, 43.18, 43.74, 44.28, 44.82, 45.35,
          45.87, 46.38, 46.88, 47.36, 47.84, 48.30, 48.75, 49.19, 49.61,
          50.02, 50.41, 50.80, 51.16, 51.51, 51.85, 52.17, 52.47, 52.75,
          53.02, 53.27, 53.50, 53.71, 53.90, 54.08, 54.23, 54.36, 54.48,
          54.57, 54.64, 54.69, 54.73, 54.74
         ]
    ax.plot(x, y, "k-", transform=ccrs.PlateCarree())

    x = [-30.00,-28.41,-26.82,-25.24,-23.65,-22.08,-20.51,-18.96,-17.41,
         -15.88,-14.36,-12.85,-11.36, -9.89, -8.43, -6.99, -5.57, -4.17,
          -2.79, -1.44, -0.10,  1.22,  2.51,  3.79,  5.04,  6.27,  7.48,
           8.67,  9.83, 10.98, 12.10, 13.21, 14.29, 15.35, 16.40, 17.42,
          18.43, 19.42, 20.39, 21.34, 22.27, 23.19, 24.10, 24.98, 25.86,
          26.71, 27.55, 28.38, 29.20, 30.00
         ]
    y = [-54.74,-54.73,-54.69,-54.64,-54.57,-54.48,-54.36,-54.23,-54.08,
         -53.90,-53.71,-53.50,-53.27,-53.02,-52.75,-52.47,-52.17,-51.85,
         -51.51,-51.16,-50.80,-50.41,-50.02,-49.61,-49.19,-48.75,-48.30,
         -47.84,-47.36,-46.88,-46.38,-45.87,-45.35,-44.82,-44.28,-43.74,
         -43.18,-42.62,-42.04,-41.46,-40.87,-40.27,-39.67,-39.06,-38.44,
         -37.82,-37.19,-36.55,-35.91,-35.26
         ]
    ax.plot(x, y, "k-", transform=ccrs.PlateCarree())

    # Plot source-type
    ax.plot(gamma,delta,"ro",markersize=10,markeredgecolor="k",transform=ccrs.PlateCarree())

    outfile = "lune.%s"%format
    fig.savefig(outfile,format=format,bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


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