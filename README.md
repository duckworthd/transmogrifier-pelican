# Transmogrifier+Pelican

Pelican-based version of [http://duckworthd.github.com][home].

Makes use of a variation of [svbtle's theme][svbtle] for its theme and
[python-markdown-mathjax][pmm] + [MathJax][mj] for rendering equations.

Portrait photo by [Happy Holly Days][hhh].

## Usage

Setup

```shell
$ python3 -m virtualenv env
$ source env/bin/activate
$ python3 -m pip install -r requirements.txt
```

To serve locally on http://localhost:8000,

```shell
$ make regenerate
$ make serve
```

To publish,

```shell
$ make publish
$ cp -r ./output/* ../duckworthd.github.io/
```

To update the theme's CSS,

```shell
$ brew install npm
$ npm install less -g
$ cd themes/svbtle/static/css
$ lessc style.less style.css
```

[home]: https://github.com/mayoff/python-markdown-mathjax
[svbtle]: https://github.com/wting/pelican-svbtle
[pmm]: https://github.com/mayoff/python-markdown-mathjax
[mj]: http://www.mathjax.org/
[hhh]: https://www.facebook.com/happyhollydayphotography
