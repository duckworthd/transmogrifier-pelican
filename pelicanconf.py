#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR   = u'Daniel Duckworth'
SITENAME = u'Strongly Convex'
SITEURL  = 'http://localhost:8000'

TIMEZONE = 'US/Pacific'

DEFAULT_LANG = u'en'

DISPLAY_PAGES_ON_MENU = False

# Feed generation is usually not desired when developing
FEED_ALL_ATOM         = None
CATEGORY_FEED_ATOM    = None
TRANSLATION_FEED_ATOM = None

LINKS = [
  ('Blog', "/"),
  ('About', "/about.html"),
]

SOCIAL = [
  ('twitter', 'http://twitter.com/duck'),
  ('github' , 'http://github.com/duckworthd'),
]

STATIC_PATHS = [
  "assets",
  "CNAME",
]

# the URL format used for articles and pages
ARTICLE_URL = ARTICLE_SAVE_AS = 'blog/{slug}.html'
PAGE_URL    = PAGE_SAVE_AS    = '{slug}.html'

# 5 articles per page
DEFAULT_PAGINATION = 5

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True

# for prettiness...
THEME = "themes/svbtle"

# for code highlighting
MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.codehilite': {'css_class': 'highlight'},
        'markdown.extensions.extra': {},
        'markdown.extensions.meta': {},
        # 'mdx_mathjax': {},
    },
    'output_format': 'html5',
}
