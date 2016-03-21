#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

# This file is only used if you use `make publish` or
# explicitly specify it as your config file.

import os
import sys
sys.path.append(os.curdir)
from pelicanconf import *

SITEURL = ''
RELATIVE_URLS = False

# Temporarily disable feeds.
#
# TODO(duckworthd): After migrating to stronglyconvex.com, set this again.
FEED_DOMAIN = None
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None

# Following items are often useful when publishing

#DISQUS_SITENAME = ""
#GOOGLE_ANALYTICS = ""
