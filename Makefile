PROJECT ?= imgfilter

CC   ?= gcc

# ---------- Directories ----------
SRC_DIR ?= src
EXT_DIR ?= $(SRC_DIR)/external
OBJ_DIR ?= obj
BIN_DIR ?= bin

# ---------- CUDA location ----------
# Try to auto-detect CUDA from nvcc in PATH first, then fall back to common locations
ifndef CUDA_HOME
  NVCC_PATH := $(shell which nvcc 2>/dev/null)
  ifneq ($(NVCC_PATH),)
    CUDA_HOME := $(shell dirname $(shell dirname $(NVCC_PATH)))
  else
    CUDA_HOME := $(firstword \
      $(wildcard /opt/cuda) \
      $(wildcard /usr/local/cuda) \
    )
  endif
endif

NVCC ?= $(if $(CUDA_HOME),$(CUDA_HOME)/bin/nvcc,nvcc)

# Build the GPU backend when nvcc is found; `make CUDA=0` forces a CPU-only
# build (gpu_none.c replaces gpu.cu and -g reports that CUDA is unavailable).
CUDA ?= $(if $(shell command -v $(NVCC) 2>/dev/null),1,0)

# ---------- CUDA architecture ----------
#
# Native SASS (sm_XX) is compiled for the detected/selected GPU, so that the
# kernel does not depend on the driver's PTX JIT.
#
# Usage:
#   make                  # auto-detect GPU, fallback to T4 (sm_75)
#   make GPU_ARCH=86      # override (e.g., RTX 3060)
#   make GPU_ARCHES="75 86"  # build fat binary for multiple GPUs

GPU_ARCH_DETECTED := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')
GPU_ARCH ?= $(if $(GPU_ARCH_DETECTED),$(GPU_ARCH_DETECTED),75)
GPU_ARCHES ?= $(GPU_ARCH)

NVCC_GENCODE :=
$(foreach arch,$(GPU_ARCHES),$(eval NVCC_GENCODE += -gencode arch=compute_$(arch),code=sm_$(arch)))

# ---------- Flags ----------
CPPFLAGS ?=
CFLAGS   ?= -Wall -Wextra -Wpedantic -O3 -fopenmp
LDFLAGS  ?= -fopenmp
LDLIBS   ?= -lm

# Vendored third-party code (stb): optimized, but not held to our warnings
EXT_CFLAGS ?= -O3 -w

# CUDA: no fused multiply-add (and no fast math), so that the GPU performs
# exactly the CPU's floating-point operations and the outputs are identical.
NVCCFLAGS ?= -O3 -fmad=false $(NVCC_GENCODE)

# Add include paths
CPPFLAGS += -I$(SRC_DIR)

# Dependency generation
DEPFLAGS := -MMD -MP

# ---------- Sources ----------
C_SRCS   := $(wildcard $(SRC_DIR)/*.c)
CU_SRCS  := $(wildcard $(SRC_DIR)/*.cu)
EXT_SRCS := $(wildcard $(EXT_DIR)/*.c)

# gpu.cu with CUDA, its stub gpu_none.c without
ifeq ($(CUDA),1)
  C_SRCS := $(filter-out $(SRC_DIR)/gpu_none.c,$(C_SRCS))
else
  CU_SRCS :=
endif

C_OBJS   := $(C_SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
CU_OBJS  := $(CU_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.cu.o)
EXT_OBJS := $(EXT_SRCS:$(EXT_DIR)/%.c=$(OBJ_DIR)/external/%.o)
OBJS     := $(C_OBJS) $(CU_OBJS) $(EXT_OBJS)

# Link with nvcc when CUDA code is present (it adds the CUDA runtime)
ifeq ($(CUDA),1)
  LINK       := $(NVCC)
  LINK_FLAGS := $(NVCC_GENCODE) -Xcompiler=-fopenmp $(LDLIBS)
else
  LINK       := $(CC)
  LINK_FLAGS := $(LDFLAGS) $(LDLIBS)
endif

TARGET ?= $(BIN_DIR)/$(PROJECT)

# The generated .d files contain rules; without this, the first of them
# would become the default goal and a plain `make` would stop relinking.
.DEFAULT_GOAL := all

DEPS := $(OBJS:.o=.d)
-include $(DEPS)

# ---------- Pretty output (optional colors) ----------
PRINTF ?= printf
ifeq ($(NO_COLOR),1)
  COLOR_RESET   :=
  COLOR_BOLD    :=
  COLOR_GREEN   :=
  COLOR_YELLOW  :=
  COLOR_BLUE    :=
  COLOR_MAGENTA :=
  COLOR_CYAN    :=
else
  COLOR_RESET   := \033[0m
  COLOR_BOLD    := \033[1m
  COLOR_GREEN   := \033[1;32m
  COLOR_YELLOW  := \033[1;33m
  COLOR_BLUE    := \033[1;34m
  COLOR_MAGENTA := \033[1;35m
  COLOR_CYAN    := \033[1;36m
endif

# ---------- Rules ----------
.PHONY: all clean rebuild help
all: $(TARGET)

$(BIN_DIR) $(OBJ_DIR) $(OBJ_DIR)/external:
	@mkdir -p $@

$(TARGET): $(OBJS) | $(BIN_DIR)
	@$(PRINTF) "$(COLOR_GREEN)Linking:$(COLOR_RESET) %s $(if $(filter 1,$(CUDA)),(CUDA sm: $(GPU_ARCHES)),(CPU only))\n" "$@"
	@$(LINK) -o $@ $(OBJS) $(LINK_FLAGS)
	@$(PRINTF) "$(COLOR_CYAN)Build complete!$(COLOR_RESET)\n"

# C objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	@$(PRINTF) "$(COLOR_BLUE)Compiling C:$(COLOR_RESET) %s\n" "$<"
	@$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

# CUDA objects
$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@$(PRINTF) "$(COLOR_MAGENTA)Compiling CUDA:$(COLOR_RESET) %s\n" "$<"
	@$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -c $< -o $@

# Third-party objects
$(OBJ_DIR)/external/%.o: $(EXT_DIR)/%.c | $(OBJ_DIR)/external
	@$(PRINTF) "$(COLOR_MAGENTA)Compiling external:$(COLOR_RESET) %s\n" "$<"
	@$(CC) $(CPPFLAGS) $(EXT_CFLAGS) $(DEPFLAGS) -c $< -o $@

clean:
	@$(PRINTF) "$(COLOR_YELLOW)Cleaning...$(COLOR_RESET)\n"
	@rm -rf $(OBJ_DIR) $(BIN_DIR)
	@$(PRINTF) "$(COLOR_GREEN)✓ Clean complete$(COLOR_RESET)\n"

rebuild: clean all

help:
	@$(PRINTF) "\n"
	@$(PRINTF) "$(COLOR_BOLD)$(COLOR_BLUE)imgfilter — Parallel Image Filters$(COLOR_RESET)\n\n"
	@$(PRINTF) "$(COLOR_BOLD)Targets:$(COLOR_RESET)\n"
	@$(PRINTF) "  $(COLOR_CYAN)all$(COLOR_RESET)       Build (default)\n"
	@$(PRINTF) "  $(COLOR_CYAN)clean$(COLOR_RESET)     Remove build artifacts\n"
	@$(PRINTF) "  $(COLOR_CYAN)rebuild$(COLOR_RESET)   Clean and rebuild\n"
	@$(PRINTF) "  $(COLOR_CYAN)help$(COLOR_RESET)      Show this message\n\n"
	@$(PRINTF) "$(COLOR_BOLD)Overrides:$(COLOR_RESET)\n"
	@$(PRINTF) "  make CC=clang\n"
	@$(PRINTF) "  make CUDA=0              # CPU-only build\n"
	@$(PRINTF) "  make GPU_ARCH=86         # CUDA architecture (default: detected, else 75)\n"
	@$(PRINTF) "  make NO_COLOR=1\n\n"
	@$(PRINTF) "$(COLOR_BOLD)Usage:$(COLOR_RESET)\n"
	@$(PRINTF) "  ./$(TARGET) [-o output] [-f format] [-t threads] [-B batch] [-b bench.json [-n trials] [-w wtrials]] [-p] [-d [-g] ...] <input>\n\n"
