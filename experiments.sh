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
  for i in {1..2}
  do
    eval "$command" > "$dir_results/$prefix/$i"
  done
}

function run_replications_no_calibrate
{
  command=$1 # calibrate
  prefix=$2

  (cd $dir_results && mkdir -p $prefix)
  for i in {1..2}
  do
    eval "$command" > "$dir_results/$prefix/$i"
  done
}

function starpu_gpu
{
  (cd $dir_starpu && make clean && make)
  for n in 19 20
  do
    prefix="g6.16xlarge.starpu_gpu-$n"
    run="$dir_starpu/nbody $n"
    run_replications "$run" "$prefix"
  done
}

function openmp_gpu
{
  (cd $dir_openmp && make clean && make)
  for n in 19 20
  do
    prefix="g6.16xlarge.openmp_gpu-$n"
    run="OMP_NUM_THREADS=32 $dir_openmp/nbody $n"
    run_replications_no_calibrate "$run" "$prefix"
  done
}

# openmp_gpu
starpu_gpu