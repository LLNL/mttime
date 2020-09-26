# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-814839
# author: Andrea Chiang (andrea4@llnl.gov)
"""
tdmtpy command line utility
"""

from argparse import ArgumentParser

from tdmtpy import __version__
from tdmtpy.configure import Configure
from tdmtpy.inversion import Inversion

def main(argv=None):
    """
    Perform inversion

    :param argv: inputs
    """
    parser = ArgumentParser(prog="tdmtpy-run", description=__doc__.strip())
    parser.add_argument("-V", "--version", action="version", version="%(prog)s"+__version__)
    parser.add_argument("file", nargs="?", default="mtinv.in",
                        help="Input parameter file, default is mtinv.in")

    args = parser.parse_args(argv)

    config = Configure(path_to_file=args.file)
    inv = Inversion(config=config)
    inv.invert()
    inv.write()


if __name__ == '__main__':
    main()
