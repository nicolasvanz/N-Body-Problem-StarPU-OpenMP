source const.sh

function replace
{
	dir=$1
	oldstr=$2
	newstr=$3

	cecho $GREEN "[+] Replacing \"$oldstr\" -> \"$newstr\""

	find $dir \( -type d -name .git -prune \) -o -type f -print0 \
		| xargs -0 $SED -i "s/$oldstr/$newstr/g"
}

function run_replications
{
  command=$1 # calibrate
  prefix=$2

  (cd $dir_results && mkdir -p $prefix)
  eval "$command" #calibrate
  for i in {1..1}
  do
    eval "$command" > "$dir_results/$prefix/$i"
  done
}

function starpu_gpu
{
  (cd $dir_starpu && make clean && make)
  for ((n=nbodies_initial_index; n<=nbodies_final_index; n++));
  do
    prefix="starpu_gpu-$n"
    run="STARPU_NCPU=0 $dir_starpu/nbody $n"
    run_replications "$run" "$prefix"
  done
}

function starpu_cpu
{
  (cd $dir_starpu && make clean && make)
  for n in 18 19
  do
    prefix="c5n.starpu_cpu-$n"
    run="$dir_starpu/nbody $n"
    run_replications "$run" "$prefix"
  done
}

function starpu_cpu_gpu
{
  starpu_parts_macro="#define PARTS"
  starpu_parts_default="$starpu_parts_macro 1"
  replace "$dir_starpu" "$starpu_parts_default" "$starpu_parts_macro $experiments_starpu_parts"
  (cd $dir_starpu && make clean && make)
  for ((n=nbodies_initial_index; n<=nbodies_final_index; n++));
  do
    prefix="starpu_cpu_gpu-$n"
    run="$dir_starpu/nbody $n"
    run_replications "$run" "$prefix"
  done
  replace "$dir_starpu" "$starpu_parts_macro $experiments_starpu_parts" "$starpu_parts_default"
}

function openmp_cpu
{
  (cd $dir_openmp && make clean && make)
  for ((n=nbodies_initial_index; n<=nbodies_final_index; n++));
  do
    prefix="openmp_cpu-$n"
    run="OMP_NUM_THREADS=$experiments_omp_threads $dir_openmp/nbody $n"
    run_replications "$run" "$prefix"
  done
}

function openmp_gpu
{
  openmp_bodyforce_use_cpu_macro="#define BODYFORCE_USE_CPU"
  openmp_bodyforce_use_cpu_default="$openmp_bodyforce_use_cpu_macro 1"
  openmp_integratepositions_use_cpu_macro="#define INTEGRATEPOSITIONS_USE_CPU"
  openmp_integratepositions_use_cpu_default="$openmp_integratepositions_use_cpu_macro 1"
  replace "$dir_openmp" "$openmp_bodyforce_use_cpu_default" "$openmp_bodyforce_use_cpu_macro 0"
  replace "$dir_openmp" "$openmp_integratepositions_use_cpu_default" "$openmp_integratepositions_use_cpu_macro 0"
  (cd $dir_openmp && make clean && make)
  for ((n=nbodies_initial_index; n<=nbodies_final_index; n++));
  do
    prefix="openmp_gpu-$n"
    run="OMP_NUM_THREADS=$experiments_omp_threads $dir_openmp/nbody $n"
    run_replications "$run" "$prefix"
  done
  replace "$dir_openmp" "$openmp_bodyforce_use_cpu_macro 0" "$openmp_bodyforce_use_cpu_default"
  replace "$dir_openmp" "$openmp_integratepositions_use_cpu_macro 0" "$openmp_integratepositions_use_cpu_default" 
}

function openmp_cpu_gpu
{
  openmp_bodyforce_use_cpu_macro="#define BODYFORCE_USE_CPU"
  openmp_bodyforce_use_cpu_default="$openmp_bodyforce_use_cpu_macro 1"
  replace "$dir_openmp" "$openmp_bodyforce_use_cpu_default" "$openmp_bodyforce_use_cpu_macro 0"
  (cd $dir_openmp && make clean && make)
  for ((n=nbodies_initial_index; n<=nbodies_final_index; n++));
  do
    prefix="openmp_cpu_gpu-$n"
    run="OMP_NUM_THREADS=$experiments_omp_threads $dir_openmp/nbody $n"
    run_replications "$run" "$prefix"
  done
  replace "$dir_openmp" "$openmp_bodyforce_use_cpu_macro 0" "$openmp_bodyforce_use_cpu_default"
}

starpu_cpu