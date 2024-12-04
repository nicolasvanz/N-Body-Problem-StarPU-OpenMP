if [[ "$OSTYPE" == "darwin"* ]]; then
	SED="gsed"
else
	SED="sed"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

function cecho
{
	echo -e "$1$2$NC"
}

dir_current=$PWD
dir_results=$dir_current/results
dir_results_raw=$dir_results/raw
dir_starpu=$dir_current/src/starpu
