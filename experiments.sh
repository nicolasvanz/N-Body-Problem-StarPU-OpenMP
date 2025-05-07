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
  for i in {1..3}
  do
    eval "$command" > "$dir_results/$prefix/$i"
  done
}

function starpu_gpu
{
  (cd $dir_starpu && make clean && make)
  for n in 18 19
  do
    prefix="starpu_gpu-$n"
    run="mpirun --hostfile hostfile -map-by slot:PE=8 $dir_starpu/nbody $n"
    scp src/starpu/nbody ec2-user@10.0.0.11:/home/ec2-user/N-Body-Problem-StarPU-OpenMP/src/starpu/
    scp src/starpu/nbody ec2-user@10.0.0.12:/home/ec2-user/N-Body-Problem-StarPU-OpenMP/src/starpu/
    scp src/starpu/nbody ec2-user@10.0.0.13:/home/ec2-user/N-Body-Problem-StarPU-OpenMP/src/starpu/
    scp src/starpu/nbody ec2-user@10.0.0.14:/home/ec2-user/N-Body-Problem-StarPU-OpenMP/src/starpu/
    scp src/starpu/nbody ec2-user@10.0.0.15:/home/ec2-user/N-Body-Problem-StarPU-OpenMP/src/starpu/
    scp src/starpu/nbody ec2-user@10.0.0.16:/home/ec2-user/N-Body-Problem-StarPU-OpenMP/src/starpu/
    scp src/starpu/nbody ec2-user@10.0.0.17:/home/ec2-user/N-Body-Problem-StarPU-OpenMP/src/starpu/
    run_replications "$run" "$prefix"
  done
}

function starpu_cpu
{
  starpu_parts_macro="#define PARTS"
  starpu_parts_default="$starpu_parts_macro 1"
  replace "$dir_starpu" "$starpu_parts_default" "$starpu_parts_macro $experiments_starpu_parts"
  (cd $dir_starpu && make clean && make)
  for ((n=nbodies_initial_index; n<=nbodies_final_index; n++));
  do
    prefix="starpu_cpu-$n"
    run="STARPU_NCUDA=0 $dir_starpu/nbody $n"
    run_replications "$run" "$prefix"
  done
  replace "$dir_starpu" "$starpu_parts_macro $experiments_starpu_parts" "$starpu_parts_default"
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

starpu_gpu
