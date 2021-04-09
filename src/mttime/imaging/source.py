# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-814839
# author: Andrea Chiang (andrea@llnl.gov)
"""
Plotting routines for mttime

.. warning::

   This module should NOT be used directly instead used the class method
   :meth:`~mttime.inversion.Inversion.plot`.
"""

import warnings
import numpy as np

from matplotlib.gridspec import GridSpec
from matplotlib import transforms
from matplotlib.path import Path as mpath
from obspy.geodetics.base import kilometers2degrees
from .beachball import beach

from mttime.utils import CARTOPY_VERSION
if CARTOPY_VERSION and CARTOPY_VERSION >= [0, 17, 0]:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
else:
    HAS_CARTOPY = False

if not HAS_CARTOPY:
    msg = "Cartopy is not installed, map plots will not work."
    warnings.warn(msg)


# Figure defaults
mm = 1/25.4 # mm in inches
#ppi = 72 # 72 pts per inch
#dpi = 300
#LINEWIDTH = 2.5/(dpi/ppi)


def _adjust_spines(ax, spines):
    """
    Adjust axis spine for the standard beach ball/waveform plot

    Function to customize figure axes.

    :param ax: the axes instance containing the spine.
    :type ax: :class:`~matplotlib.axes.Axes`
    :param spines: spine type.
    :type spines: list(str)
    """
    ax.tick_params(direction='in')
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
    else:
        ax.xaxis.set_ticks([])


def new_page(nsta, nrows, ncols, title, annot=None, offset=2):
    """
    Creates a new figure

    Creates a new :class:`~matplotlib.figure.Figure` and customizes
    subplot layout and figure axes. This is the figure layout for plotting
    focal mechanisms and waveform fits.

    :param nsta: number of stations to plot.
    :type nsta: int
    :param nrows: number of rows to per page.
    :type nrows: int
    :param ncols: number of columns per page.
    :type ncols: int
    :param title: title for each column.
    :type title: list of str
    :param annot: figure annotations.
    :type annot: str
    :param offset: offset between waveform traces.
    :type offset: int or float
    :return: figure container, axis of the focal mechanism plot, and axes for the waveform traces.
    :rtype: class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`, list
    """
    import matplotlib.pyplot as plt

    if annot is None:
        annot = ""

    if nsta > nrows:
        raise ValueError("Maximum number of stations per page is %d."%nrows)

    gs = GridSpec(nrows+offset, ncols, hspace=0.6, wspace=0.15)
    f = plt.figure(figsize=(190*mm, 230*mm))
    
    # Annotations and beach balls
    ax0 = f.add_subplot(gs[0:offset,:], xlim=(-5,3.55), ylim=(-0.75,0.6), aspect="equal")
    ax0.text(-5, 0, annot, verticalalignment='center')
    ax0.set_axis_off()

    # Waveforms
    ax1 = np.empty((nsta,3), dtype=np.object) # create empty axes
    for i in range(nsta):
        ax1[i,0] = f.add_subplot(gs[i+offset, 0])
        ax1[i,1] = f.add_subplot(gs[i+offset, 1])
        ax1[i,2] = f.add_subplot(gs[i+offset, 2])

    # Adjust axes
    for i in range(nsta-1):
        _adjust_spines(ax1[i,0],['left'])
        _adjust_spines(ax1[i,1], [])
        _adjust_spines(ax1[i,2], ["bottom"])
    _adjust_spines(ax1[-1,0],['left',"bottom"])
    _adjust_spines(ax1[-1,1],['bottom'])
    _adjust_spines(ax1[-1,2],['bottom'])
    
    # Title
    for i in range(ncols):
        ax1[0,i].set_title(title[i], verticalalignment='bottom', pad=15)
    
    return (f, ax0, ax1)


def plot_waveform_fits(tensor, show, format, nrows=10):
    """
    Plot waveform fits and focal mechanisms

    :param tensor:
    :param nrows: maximum stations per page. Default is 10.
    :type nrows: int
    """
    import matplotlib.pyplot as plt

    # Number decimals to print
    dist = ("{0:.0f}")

    # Set page layout
    nsta = len(tensor.station_table.index)
    ncols = len(tensor.components)
    a = nsta / nrows
    nPages = int(a) + ((int(a) - a) != 0)
    lst = list(range(0, nsta, nrows))
    lst.append(nsta)
    pages = (range(lst[i], lst[i + 1]) for i in range(nPages))

    # Brief summary of inversion
    annot = tensor._get_summary()

    # Decompositions to plot
    if tensor.inversion_type == "Deviatoric":
        fm_title = ["Deviatoric", "DC", "CLVD"]
        fm = [tensor._m, tensor.fps[0], tensor.clvd]
        fm_width = [1, 0.01*tensor.pdc, 0.01*tensor.pclvd]
    elif tensor.inversion_type == "Full":
        fm_title = ['Full', 'ISO', 'Deviatoric']
        fm = (tensor._m, tensor.iso, tensor.dev)
        fm_width = [1, 0.01*tensor.piso, 0.01*(tensor.pdc+tensor.pclvd)]
    fm_sign = 1 - 0.25 * fm_width[1], 2.5 - 0.5 * fm_width[2]

    # Station location around beach ball
    x = 0.55 * np.sin(tensor.station_table.azimuth * np.pi / 180).values
    y = 0.55 * np.cos(tensor.station_table.azimuth * np.pi / 180).values
    tri_color = np.array(np.repeat("0.5",nsta), dtype="<U5")
    tri_color[tensor.station_table[tensor.components].sum(axis=1) > 0] = "green"

    # Waveform line colors
    syntcol = np.empty(tensor.station_table[tensor.components].shape, dtype='<U5')
    syntcol[tensor.station_table[tensor.components] == 1] = 'green'
    syntcol[tensor.station_table[tensor.components] == 0] = '0.5'
    datacol = "black"

    for page, group in enumerate(pages):
        f, ax0, ax1 = new_page(len(group), nrows+1, ncols, tensor.components, annot=annot)
        # Plot beach balls
        for i in range(len(fm)):
            beach1 = beach(fm[i], xy=(i + 0.5 * i, 0), width=fm_width[i], show_iso=True)
            ax0.add_collection(beach1)
            ax0.text(i + 0.5 * i, 0.55, fm_title[i], horizontalalignment='center')
        ax0.text(fm_sign[0], 0, '=', horizontalalignment='center', verticalalignment='center')
        ax0.text(fm_sign[1], 0, '+', horizontalalignment='center', verticalalignment='center')
        # Plot stations around beach ball
        for xi,yi,azi,col in zip(x, y, tensor.station_table.azimuth, tri_color):
            ax0.plot(xi, yi, marker=(3, 0, -1*azi), color=col, zorder=101, markersize=5)
        # Plot waveforms
        for i, stat in enumerate(group):
            t = np.arange(
                0,
                tensor.station_table.npts[stat] * tensor.station_table.dt[stat],
                tensor.station_table.dt[stat]
            )
            data = tensor._data[stat]
            synt = tensor._synthetics[stat]
            ymin = np.min([data, synt])
            ymax = np.max([data, synt])
            for j in range(len(tensor.components)):
                ax1[i, j].plot(t, data[:, j], color=datacol, clip_on=False)
                ax1[i, j].plot(t, synt[:, j], color=syntcol[stat, j], dashes=[6, 2], clip_on=False)
                ax1[i, j].set_ylim(ymin, ymax)
                ax1[i, j].set_xlim(0, t[-1])
            # Set ticks and labels
            ax1[i, 0].set_yticks([ymin, 0, ymax])
            ax1[i, 0].set_yticklabels(['%.2e' % ymin, '0', '%.2e' % ymax])
            # Station name, distance and azimuth
            dist = dist.format(tensor.station_table.distance[stat])
            label = '\n'.join(
                [tensor.station_table.station[stat],
                 r'$\Delta,\theta$=%s,%-.0f' % (dist, tensor.station_table.azimuth[stat])]
            )
            ax1[i, 0].text(0,ymax, label, verticalalignment="bottom")
            # Sample shift and VR
            ax1[i, 1].text(
                t[-1], ymax,
                'ts,VR=%d,%.0f'%(tensor.station_table.ts[stat], tensor.station_table.VR[stat]),
                horizontalalignment="right",
                verticalalignment="bottom"
            )
        # Label last row only
        for column in range(3):
            ax1[i, column].set_xlabel('Time [s]')

        if show:
            plt.show()
        else:
            outfile = "bbwaves.d%07.4f.%02d.%s" % (tensor.depth, page, format)
            f.savefig(outfile, format=format, transparent=True)
            plt.close(f)


def beach_map(m, event, longitude, latitude, distance, used, show, format):
    """
    Plot beach ball on a map

    Function to plot stations and focal mechanisms on a Stamen terrain background.

    :param m: focal mechanisms, refer to :func:`~tdmtpy.image.beach` function
        for the supported formats.
    :type m: list
    :param event: event origin time, longitude and latitude, refer to :class:`~tdmtpy.configure.Configure`
        for details.
    :type event: dict
    :param longitude: station longitudes.
    :type longitude: list or :class:`~numpy.ndarray`
    :param latitude: station latitudes.
    :type latitude: list or :class:`~numpy.ndarray`
    :param distance: source-receiver distance.
    :type distance: list or :class:`~numpy.ndarray`
    :param used: inverted stations. Sets the station marker colors,
        ``True`` for green and ``False`` for gray.
    :type used: list or :class:`~numpy.ndarray`
    :param show: Turn on interactive display.
    :type show: bool
    :param format: figure file format.
    :type format: str
    """
    import matplotlib.pyplot as plt

    # Calculate image extent based on epicentral distance
    width = kilometers2degrees(0.5 * max(distance))
    height = kilometers2degrees(0.5 * max(distance))

    lat1 = min(latitude) - height
    lat2 = max(latitude) + height
    lon1 = min(longitude) - width
    lon2 = max(longitude) + width

    data_crs = ccrs.PlateCarree()
    evlo = event["longitude"]
    evla = event["latitude"]
    point = (evlo,evla)

    projection = ccrs.PlateCarree()

    fig = plt.figure(figsize=(95*mm, 115*mm))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_extent([lon1, lon2, lat1, lat2])

    # Add borders and coastline
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

    # Add tick labels
    gl = ax.gridlines(crs=data_crs, draw_labels=True)
    gl.top_labels = False
    #gl.xlabel_style = {"size": smaller}
    #gl.ylabel_style = {"size": SMALL_SIZE}

    # Plot stations
    colors = [ "green" if i else "0.5" for i in used]
    for i in range(len(colors)):
        ax.plot(
            longitude[i],
            latitude[i],
            marker="^",
            color=colors[i],
            markeredgecolor="black",
            transform=data_crs
        )

    # Plot beach ball on map
    x, y = projection.transform_point(*point, src_crs=data_crs)
    bb = beach(m, xy=(x, y), facecolor="red", width=0.25, show_iso=True)
    ax.add_collection(bb)

    # Add title
    ax.set_title(event["datetime"])

    if show:
        plt.show()
    else:
        outfile = "map.%s" % format
        fig.savefig(outfile, format=format, transparent=True)
        plt.close(fig)


def plot_lune(gamma,delta,show,format):
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
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(95*mm, 115*mm))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertAzimuthalEqualArea())

    # Draw boundary around the lune
    longitude = np.concatenate([[-30], np.tile(-30, 180), [-30], np.tile(30, 180), [30]])
    latitude = np.concatenate([[-90], np.arange(-90, 90, 1), [90], np.arange(90, -90, -1), [-90]])
    codes = np.hstack([mpath.MOVETO, [mpath.LINETO] * 180,
                       mpath.MOVETO, [mpath.LINETO] * 180,
                       mpath.MOVETO])
    verts = np.column_stack([longitude[::-1], latitude[::-1]])
    path = mpath(verts, codes[::-1])
    ax.set_boundary(path, transform=ccrs.PlateCarree())
    ax.set_extent([-30, 30, -90, 90], ccrs.PlateCarree())

    # Add gridlines
    xlocs = np.arange(-30, 30 + 10, 10, dtype=np.int)
    ylocs = np.arange(-90, 90 + 10, 10, dtype=np.int)
    ax.gridlines(xlocs=xlocs, ylocs=ylocs)

    # Offset text by mm
    text_transform = ccrs.PlateCarree()._as_mpl_transform(ax)
    inches = 1.5
    left = transforms.offset_copy(text_transform, fig, units="inches", x=-1*inches*mm)
    right = transforms.offset_copy(text_transform, fig, units="inches", x=inches*mm)
    top = transforms.offset_copy(text_transform, fig, units="inches", y=inches*mm)
    bottom = transforms.offset_copy(text_transform, fig, units="inches", y=-1*inches*mm)

    # Theoretical sources
    x = [0, -30, -30, 0, 30, 30, -30, 30, 0]
    y = [-90, 0, 35.2644, 90, 0, -35.2644, 60.5038, -60.5038, 0]
    sources = ["-ISO", "+CLVD", "+LVD", "+ISO", "-CLVD", "-LVD", "+Crack", "-Crack", "DC"]
    offset = [bottom, left, left, top, right, right, left, right, bottom]

    halign = ["center", "right", "right", "center", "left", "left", "right", "left", "center"]
    valign = ["top", "center", "center", "bottom", "center", "center", "center", "center", "top"]
    ax.plot(x, y, "o", color="black", transform=ccrs.PlateCarree(), clip_on=False)
    for i in range(len(x)):
        ax.text(x[i], y[i], sources[i],
                transform=offset[i], clip_on=False,
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
    ax.plot(gamma, delta, "ro", markeredgecolor="k", transform=ccrs.PlateCarree())

    if show:
        plt.show()
    else:
        outfile = "lune.%s"%format
        fig.savefig(outfile, format=format, transparent=True)
        plt.close(fig)


def beach_mw_depth(tensors, event, show, format):
    """
    Plot depth search result

    A function that plots focal mechanism, variance reduction, and moment magnitude
    as a function of source depth.

    :param tensors: moment tensor solutions at various depths.
    :type tensors: a list of :class:`~tdmtpy.tensor.Tensor`
    :param event: event origin time, longitude and latitude, refer to :class:`~tdmtpy.configure.Configure`
        for details.
    :type event: dict
    :param show: Turn on interactive display.
    :type show: bool
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(115*mm, 95*mm))
    ax1 = fig.add_subplot(1, 1, 1)

    # Axis 1 - variance reduction
    color1 = "red"
    ax1.set_xlabel("Depth [km]")
    ax1.set_ylabel("VR [%]", color=color1)
    ax1.set_ylim(0, 120)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_yticklabels([0, 20, 40, 60, 80, 100])
    title = "%s | %5.2f %5.2f"%(event["datetime"],event["longitude"],event["latitude"])
    ax1.set_title(title)

    color2 = "mediumblue"
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel("Mw", color=color2)
    source_depth = [tensor.depth for tensor in tensors]
    margin = (max(source_depth)-min(source_depth))/6
    #if margin == 0:
    #    margin = 1.0
    ax2.set_xlim([min(source_depth)-margin, max(source_depth)+margin])
    ax2.set_ylim(0,12)
    ax2.set_yticks([0, 2, 4, 6, 8, 10])
    ax2.set_yticklabels([0, 2, 4, 6, 8, 10])

    # Find solution with highest VR
    vr = [ _m.total_VR for _m in tensors ]
    ax1.vlines(tensors[np.argmax(vr)].depth, 0, 100, color="black")
    for tensor in tensors:
        ax1.plot(tensor.depth, tensor.total_VR, "o", color=color1)
        bb = beach(tensor.m,
                   xy=(0,0),
                   facecolor=color1,
                   width=0.25,
                   show_iso=True,
                  )
        # Move the beach ball to the right position
        bb.set_transform(fig.dpi_scale_trans)
        bb.set_offsets((tensor.depth, tensor.total_VR+10))
        bb._transOffset = ax1.transData
        ax1.add_collection(bb)
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2.plot(tensor.depth, tensor.mw, "o", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

    if show:
        plt.show()
    else:
        outfile = "depth.bbmw.%s" % format
        fig.savefig(outfile, format=format, transparent=True)
        plt.close(fig)
