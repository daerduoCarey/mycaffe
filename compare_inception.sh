grep -E "Iteration [0-9]*, loss" inception_6.log > inception_6
grep -E "Iteration [0-9]*, loss" inception_6_bn.log > inception_6_bn
grep -E "Iteration [0-9]*, loss" inception_6_bn_final.log > inception_6_bn_final

python stat_plot.py loss 40 inception_6 inception_6_bn inception_6_bn_final
