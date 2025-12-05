
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pedantic
CXX = g++
# Run setup-env from vulkan
# source ~/repos/vulkan/1.4.328.1/setup-env.sh

ARCH := $(shell uname -m)
VULKAN_SDK := $(HOME)/repos/vulkan/1.4.328.1/$(ARCH)
STB_DIR := $(HOME)/repos/stb
export VULKAN_SDK
export PATH := $(VULKAN_SDK)/bin:$(PATH)
export LD_LIBRARY_PATH := $(VULKAN_SDK)/lib:$(LD_LIBRARY_PATH)
export VK_ADD_LAYER_PATH := $(VULKAN_SDK)/share/vulkan/explicit_layer.d:$(VK_ADD_LAYER_PATH)
export PKG_CONFIG_PATH := $(VULKAN_SDK)/share/pkgconfig:$(VULKAN_SDK)/lib/pkgconfig:$(PKG_CONFIG_PATH)

vulkan_ink: vulkan_ink.cpp build/levelize.comp.spv.h
	$(CXX) $(CXXFLAGS) $^ -I $(VULKAN_SDK)/include/ -I $(STB_DIR) -L $(VULKAN_SDK)/lib/ -lvulkan -o $@

build/%.comp.spv: shaders/%.comp
	mkdir -p build
	glslangValidator -V $< -o $@

build/%.comp.spv.h: build/%.comp.spv
	mkdir -p build
	xxd -i $< > $@

clean:
	rm -f vulkan_ink
	rm -rf _build