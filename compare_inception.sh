USAGE="compare_inception.sh [loss | top1 | top5]"

if [ -z $1 ]; then
	echo $USAGE
	exit 1
fi

if [ "$1" == "loss" ]; then
	grep -E "Iteration [0-9]*, loss" inception_6.log > inception_6
	grep -E "Iteration [0-9]*, loss" inception_6_bn.log > inception_6_bn
	grep -E "Iteration [0-9]*, loss" inception_6_bn_final.log > inception_6_bn_final
	python stat_plot.py loss 40 inception_6 inception_6_bn inception_6_bn_final

elif [ "$1" == "top1" ]; then
	grep -E "Test net output #7" inception_6.log > inception_6
	grep -E "Test net output #7" inception_6_bn.log > inception_6_bn
	grep -E "Test net output #7" inception_6_bn_final.log > inception_6_bn_final
	python stat_plot.py loss3/top-1 4000 inception_6 inception_6_bn inception_6_bn_final

elif [ "$1" == "top5" ]; then
	grep -E "Test net output #8" inception_6.log > inception_6
	grep -E "Test net output #8" inception_6_bn.log > inception_6_bn
	grep -E "Test net output #8" inception_6_bn_final.log > inception_6_bn_final
	python stat_plot.py loss3/top-5 4000 inception_6 inception_6_bn inception_6_bn_final

else
	echo $USAGE
fi
