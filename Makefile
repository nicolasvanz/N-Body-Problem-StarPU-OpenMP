IMPL ?= openmp
PYTHON ?= python3
RTOL ?= 1e-3
ATOL ?= 1e-5
MAX_REPORT ?= 10

ifeq ($(IMPL),openmp)
SUBDIR := src/openmp
else ifeq ($(IMPL),starpu)
SUBDIR := src/starpu
else
$(error IMPL must be 'openmp' or 'starpu' (got '$(IMPL)'))
endif

.PHONY: all clean compare diff-txt help openmp starpu openmp-% starpu-% run run-openmp run-starpu

all diff-txt:
	@$(MAKE) -C $(SUBDIR) $@

compare:
	@gcc -O2 -o src/openmp/bin2txt src/bin2txt.c
	@cd src/openmp && ./bin2txt
	@$(PYTHON) src/compare.py \
		--computed src/debug/computed_pos_12 \
		--solution src/debug/solution_pos_12 \
		--kind pos \
		--rtol $(RTOL) --atol $(ATOL) --max-report $(MAX_REPORT)
	@$(PYTHON) src/compare.py \
		--computed src/debug/computed_vel_12 \
		--solution src/debug/solution_vel_12 \
		--kind vel \
		--rtol $(RTOL) --atol $(ATOL) --max-report $(MAX_REPORT)

clean:
	@$(MAKE) -C src/openmp clean
	@$(MAKE) -C src/starpu clean
	@rm -f src/debug/computed_* src/debug/*.txt
	@rm -f src/bin2txt

openmp:
	@$(MAKE) -C src/openmp all

starpu:
	@$(MAKE) -C src/starpu all

openmp-%:
	@$(MAKE) -C src/openmp $*

starpu-%:
	@$(MAKE) -C src/starpu $*

run-openmp:
	@cd src/openmp && ./nbody $(ARGS)

run-starpu:
	@cd src/starpu && ./nbody $(ARGS)

run:
	@$(MAKE) run-$(IMPL)

help:
	@echo "Usage: make IMPL={openmp|starpu} <target>"
	@echo "Common targets: all clean compare diff-txt"
	@echo "Shorthand: make openmp, make starpu, make openmp-<target>, make starpu-<target>"
	@echo "Run: make run IMPL={openmp|starpu} ARGS='<program args>'"
