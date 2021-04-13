# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-814839
# author: Andrea Chiang (andrea@llnl.gov)

import os


def _get_custom_mplstyle(file):
    style_dir = os.path.join(os.path.dirname(__file__), "styles")
    file_path = os.path.join(style_dir, file)

    return file_path


MTTIME_MPLSTYLE = _get_custom_mplstyle("mttime.mplstyle")