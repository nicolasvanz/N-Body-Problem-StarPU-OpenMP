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
    run="mpirun --hostfile hostfile -map-by slot:PE=4 $dir_starpu/nbody $n"
    for ip in $(awk '{print $1}' hostfile); do
        echo "Copying to $ip..."
        scp src/starpu/nbody ec2-user@$ip:/home/ec2-user/N-Body-Problem-StarPU-OpenMP/src/starpu/
    done
    run_replications "$run" "$prefix"
  done
}

function starpu_cpu_gpu
{
  (cd $dir_starpu && make clean && make)
  for n in 18 19
  do
    prefix="starpu_cpu_gpu-$n"
    run="mpirun --hostfile hostfile -map-by slot:PE=4 $dir_starpu/nbody $n"
    for ip in $(awk '{print $1}' hostfile); do
        echo "Copying to $ip..."
        scp src/starpu/nbody ec2-user@$ip:/home/ec2-user/N-Body-Problem-StarPU-OpenMP/src/starpu/
    done
    run_replications "$run" "$prefix"
  done
}

starpu_gpu
