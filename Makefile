PROJECT ?= imgfilter

CC   ?= gcc

# ---------- Directories ----------
SRC_DIR ?= src
EXT_DIR ?= $(SRC_DIR)/external
OBJ_DIR ?= obj
BIN_DIR ?= bin

# ---------- Flags ----------
CPPFLAGS ?=
CFLAGS   ?= -Wall -Wextra -Wpedantic -O3 -fopenmp
LDFLAGS  ?= -fopenmp
LDLIBS   ?= -lm

# Vendored third-party code (stb): optimized, but not held to our warnings
EXT_CFLAGS ?= -O3 -w

# Add include paths
CPPFLAGS += -I$(SRC_DIR)

# Dependency generation
DEPFLAGS := -MMD -MP

# ---------- Sources ----------
C_SRCS   := $(wildcard $(SRC_DIR)/*.c)
C_OBJS   := $(C_SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
EXT_SRCS := $(wildcard $(EXT_DIR)/*.c)
EXT_OBJS := $(EXT_SRCS:$(EXT_DIR)/%.c=$(OBJ_DIR)/external/%.o)
OBJS     := $(C_OBJS) $(EXT_OBJS)

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
	@$(PRINTF) "$(COLOR_GREEN)Linking:$(COLOR_RESET) %s\n" "$@"
	@$(CC) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)
	@$(PRINTF) "$(COLOR_CYAN)Build complete!$(COLOR_RESET)\n"

# C objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	@$(PRINTF) "$(COLOR_BLUE)Compiling C:$(COLOR_RESET) %s\n" "$<"
	@$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

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
	@$(PRINTF) "  make NO_COLOR=1\n\n"
	@$(PRINTF) "$(COLOR_BOLD)Usage:$(COLOR_RESET)\n"
	@$(PRINTF) "  ./$(TARGET) [-o output] [-f format] [-t threads] [-B batch] [-b bench.json [-n trials] [-w wtrials]] [-p] <input>\n\n"
