#!/usr/bin/env python

AUTHOR       = 'Daniel Duckworth'
SITENAME     = 'Strongly Convex'
SITEURL      = ''
PATH         = 'content'
TIMEZONE     = 'Europe/Rome'
DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

LINKS = [
  ('Blog', "/"),
  ('About', "/about.html"),
]

SOCIAL = [
  ('twitter', 'http://twitter.com/duck'),
  ('github' , 'http://github.com/duckworthd'),
]

# These files and folders are copied as-is.
STATIC_PATHS = [
  "assets",
  "CNAME",
]

# the URL format used for articles and pages
ARTICLE_URL = ARTICLE_SAVE_AS = 'blog/{slug}.html'
PAGE_URL    = PAGE_SAVE_AS    = '{slug}.html'

# 5 articles per page
DEFAULT_PAGINATION = 10

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
