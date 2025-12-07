
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pedantic
CXX = g++
# Run setup-env from vulkan
# https://sdk.lunarg.com/sdk/download/1.4.328.1/linux/vulkansdk-linux-x86_64-1.4.328.1.tar.xz
# source ~/repos/vulkan/1.4.328.1/setup-env.sh

# Get the directory containing this Makefile
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

ARCH := $(shell uname -m)
VULKAN_SDK := $(MAKEFILE_DIR)/external/vulkan/1.4.328.1/$(ARCH)
STB_DIR := external/stb
export VULKAN_SDK
export PATH := $(VULKAN_SDK)/bin:$(PATH)
export LD_LIBRARY_PATH := $(VULKAN_SDK)/lib:$(LD_LIBRARY_PATH)
export VK_ADD_LAYER_PATH := $(VULKAN_SDK)/share/vulkan/explicit_layer.d:$(VK_ADD_LAYER_PATH)
export PKG_CONFIG_PATH := $(VULKAN_SDK)/share/pkgconfig:$(VULKAN_SDK)/lib/pkgconfig:$(PKG_CONFIG_PATH)

$(shell which glslangValidator)


all: vulkan_ink

external/stb/stb_image.h:
	git submodule update --init

build/%.comp.spv: shaders/%.comp $(VULKAN_SDK)/bin/glslangValidator
	mkdir -p build
	glslangValidator -V $< -o $@

build/%.comp.spv.h: build/%.comp.spv
	mkdir -p build
	xxd -i $< > $@

vulkan_ink: vulkan_ink.cpp build/levelize.comp.spv.h external/stb/stb_image.h
	$(CXX) $(CXXFLAGS) $^ -I $(VULKAN_SDK)/include/ -I $(STB_DIR) -L $(VULKAN_SDK)/lib/ -lvulkan -o $@

$(VULKAN_SDK)/bin/glslangValidator:
	echo building $@
	mkdir -p external/vulkan
	cd external/vulkan && wget https://sdk.lunarg.com/sdk/download/1.4.328.1/linux/vulkansdk-linux-x86_64-1.4.328.1.tar.xz
	cd external/vulkan && tar xf vulkansdk-linux-x86_64-1.4.328.1.tar.xz


clean:
	rm -f vulkan_ink
	rm -rf _build