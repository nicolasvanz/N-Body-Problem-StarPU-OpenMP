CONTAINER1="starpu"
CONTAINER2="openmp"

docker compose run -d --name $CONTAINER1 --remove-orphans starpu
docker compose run -d --name $CONTAINER2 --remove-orphans openmp

docker exec $CONTAINER1 sh -c 'cd ../.. && bash experiments.sh starpu'
docker exec $CONTAINER2 sh -c 'cd ../.. && bash experiments.sh openmp'

docker kill $CONTAINER1 $CONTAINER2