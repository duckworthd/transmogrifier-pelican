#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR   = u'Daniel Duckworth'
SITENAME = u'STRONGLY CONVEX'
SITEURL  = ''

TIMEZONE = 'US/Pacific'

DEFAULT_LANG = u'en'

DISPLAY_PAGES_ON_MENU = False

# Feed generation is usually not desired when developing
FEED_ALL_ATOM         = None
CATEGORY_FEED_ATOM    = None
TRANSLATION_FEED_ATOM = None

LINKS = [
  ('Words', "/"),
  ('Code' , "/pages/code.html" ),
  ('About', "/pages/about.html"),
]

SOCIAL = [
  ('twitter', 'http://twitter.com/duck'),
  ('github' , 'http://github.com/duckworthd'),
]

STATIC_PATHS = [
  "assets",
]


DEFAULT_PAGINATION = False

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

# for additional functionality...
PLUGIN_PATH = 'plugins'
PLUGINS = [
  'render_math',
]

# for prettiness...
THEME = "themes/svbtle"

# for code highlighting, math
MD_EXTENSIONS = [
  'codehilite(css_class=highlight)',
  'extra',
  'mathjax',
]
import mdx_mathjax  # force loading of mathjax extension
