PY=python
PELICAN=pelican
PELICANOPTS=

BASEDIR=$(CURDIR)
INPUTDIR=$(BASEDIR)/content
OUTPUTDIR=$(BASEDIR)/output
CONFFILE=$(BASEDIR)/pelicanconf.py
PUBLISHCONF=$(BASEDIR)/publishconf.py
ENV=$(BASEDIR)/env

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	PELICANOPTS += -D
endif

help:
	@echo 'Makefile for a pelican Web site                                             '
	@echo '                                                                            '
	@echo 'Usage:                                                                      '
	@echo '   make html                        (re)generate the web site               '
	@echo '   make clean                       remove the generated files              '
	@echo '   make clean-env                   remove the generated files + virtualenv '
	@echo '   make regenerate                  regenerate files upon modification      '
	@echo '   make publish                     generate using production settings      '
	@echo '   make serve [PORT=8000]           serve site at http://localhost:8000     '
	@echo '                                                                            '
	@echo 'Set the DEBUG variable to 1 to enable debugging, e.g. make DEBUG=1 html     '
	@echo '                                                                            '

html: dependencies
	. $(ENV)/bin/activate && $(PELICAN) $(INPUTDIR) -o $(OUTPUTDIR) -s $(CONFFILE) $(PELICANOPTS)

clean:
	rm -rf $(OUTPUTDIR)
	rm -rf $(BASEDIR)/*.pyc $(BASEDIR)/cache

clean-env: clean
	rm -rf $(ENV)

regenerate: dependencies
	. $(ENV)/bin/activate && $(PELICAN) -r $(INPUTDIR) -o $(OUTPUTDIR) -s $(CONFFILE) $(PELICANOPTS)

publish: dependencies
	. $(ENV)/bin/activate && $(PELICAN) $(INPUTDIR) -o $(OUTPUTDIR) -s $(PUBLISHCONF) $(PELICANOPTS)

dependencies:
	pip install 'pip>=1.5.5' wheel virtualenv
	test -e $(ENV) || virtualenv $(ENV)
	. $(ENV)/bin/activate && pip install -r $(BASEDIR)/requirements.txt --use-wheel

serve:
ifdef PORT
	. $(ENV)/bin/activate && cd $(OUTPUTDIR) && $(PY) -m pelican.server $(PORT)
else
	. $(ENV)/bin/activate && cd $(OUTPUTDIR) && $(PY) -m pelican.server
endif

.PHONY: help html clean clean-env regenerate publish dependencies serve
