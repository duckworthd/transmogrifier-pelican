# Transmogrifier+Pelican

Pelican-based version of [http://duckworthd.github.com][home].

Makes use of a variation of [svbtle's theme][svbtle] for its theme and
[pelican-render-math][rm] + [MathJax][mj] for rendering equations.

## Usage

Setup

```shell
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
apt install npm
npm install less -g
```

To serve locally on http://localhost:8000,

```shell
invoke livereload
```

To publish,

```shell
invoke publish
cp -r ./output/* ../duckworthd.github.io/
```

To update the theme's CSS,

[home]: https://github.com/mayoff/python-markdown-mathjax
[svbtle]: https://github.com/wting/pelican-svbtle
[rm]: https://github.com/pelican-plugins/render-math
[mj]: http://www.mathjax.org/
[hhh]: https://www.facebook.com/happyhollydayphotography
