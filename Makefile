IMPL ?= openmp

ifeq ($(IMPL),openmp)
SUBDIR := src/openmp
else ifeq ($(IMPL),starpu)
SUBDIR := src/starpu
else
$(error IMPL must be 'openmp' or 'starpu' (got '$(IMPL)'))
endif

.PHONY: all clean compare compile run-bin2txt compare-txt diff-txt help openmp starpu openmp-% starpu-% run run-openmp run-starpu

all clean compare compile run-bin2txt compare-txt diff-txt:
	@$(MAKE) -C $(SUBDIR) $@

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
	@echo "Common targets: all clean compare compile run-bin2txt compare-txt diff-txt"
	@echo "Shorthand: make openmp, make starpu, make openmp-<target>, make starpu-<target>"
	@echo "Run: make run IMPL={openmp|starpu} ARGS='<program args>'"
