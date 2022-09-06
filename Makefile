PATH := ./env/bin:${PATH}
PY :=  $(VIRTUAL_ENV)/bin/python3
DOCKER :=  /usr/bin/docker


include .env
export

SRC := src
DIST := dist
BUILD := build

PWD := $(shell pwd)

.PHONY: env test all dev clean dev pyserve $(SRC) $(DIST) $(BUILD)


%: # https://www.gnu.org/software/make/manual/make.html#Automatic-Variables 
		@:


piu:
		$(PY) -m pip install --upgrade $(filter-out $@,$(MAKECMDGOALS))
		$(PY) -m pip freeze > requirements.txt

pia: requirements.txt
		$(PY) -m pip install -r requirements.txt
