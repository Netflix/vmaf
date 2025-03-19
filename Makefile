# Path and environment setup
VENV := .venv
VIRTUAL_ENV_PATH := $(VENV)/bin

# Build tools configured in the virtual environment
PYTHON_INTERPRETER := python3.10
VENV_PIP := $(VIRTUAL_ENV_PATH)/pip
VENV_PYTHON := $(VIRTUAL_ENV_PATH)/python
MESON := $(VIRTUAL_ENV_PATH)/meson
MESON_SETUP := $(MESON) setup
NINJA := $(VIRTUAL_ENV_PATH)/ninja

# Build types and options
BUILDTYPE_RELEASE := --buildtype release
BUILDTYPE_DEBUG := --buildtype debug
MESON_OPTIONS = -Denable_float=true

# Directories
LIBVMAF_DIR := libvmaf
BUILD_DIR := $(LIBVMAF_DIR)/build
DEBUG_DIR := $(LIBVMAF_DIR)/debug

.PHONY: default all debug build install cythonize clean distclean cythonize-deps
.NOTPARALLEL: clean distclean

default: build

all: build debug install test cythonize

$(BUILD_DIR)/build.ninja: $(MESON) $(NINJA)
	PATH="$(VIRTUAL_ENV_PATH):$$PATH" $(MESON_SETUP) $(BUILD_DIR) $(LIBVMAF_DIR) $(BUILDTYPE_RELEASE) $(MESON_OPTIONS)

$(DEBUG_DIR)/build.ninja: $(MESON) $(NINJA)
	PATH="$(VIRTUAL_ENV_PATH):$$PATH" $(MESON_SETUP) $(DEBUG_DIR) $(LIBVMAF_DIR) $(BUILDTYPE_DEBUG) $(MESON_OPTIONS)

cythonize: cythonize-deps
	cd python && ../$(VENV_PYTHON) setup.py build_ext --build-lib .

build: $(BUILD_DIR)/build.ninja $(NINJA)
	PATH="$(VIRTUAL_ENV_PATH):$$PATH" $(NINJA) -vC $(BUILD_DIR)

test: build $(NINJA)
	PATH="$(VIRTUAL_ENV_PATH):$$PATH" $(NINJA) -vC $(BUILD_DIR) test

debug: $(DEBUG_DIR)/build.ninja $(NINJA)
	PATH="$(VIRTUAL_ENV_PATH):$$PATH" $(NINJA) -vC $(DEBUG_DIR)

install: $(BUILD_DIR)/build.ninja $(NINJA)
	PATH="$(VIRTUAL_ENV_PATH):$$PATH" $(NINJA) -vC $(BUILD_DIR) install

clean:
	rm -rf $(BUILD_DIR) $(DEBUG_DIR)
	rm -f python/vmaf/core/adm_dwt2_cy.c*

distclean: clean
	rm -rf $(VENV)

# Set up or rebuild virtual environment
$(VENV_PIP):
	@echo "Setting up the virtual environment..."
	$(PYTHON_INTERPRETER) -m venv $(VENV) || { echo "Failed to create virtual environment"; exit 1; }
	$(VENV_PYTHON) -m pip install --upgrade pip || { echo "Failed to upgrade pip"; exit 1; }
	@echo "Virtual environment setup complete."

$(VIRTUAL_ENV_PATH)/meson: $(VENV_PIP)
	$(VENV_PIP) install meson || { echo "Failed to install meson"; exit 1; }

$(VIRTUAL_ENV_PATH)/ninja: $(VENV_PIP)
	$(VENV_PIP) install ninja || { echo "Failed to install ninja"; exit 1; }

cythonize-deps: $(VENV_PIP)
	$(VENV_PIP) install setuptools cython numpy || { echo "Failed to install dependencies"; exit 1; }