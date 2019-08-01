#!/bin/sh
if [ ! -f BSR_bsds500.tgz ]
then
	wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
fi

if [ ! -d BSR ]
then
	tar -zxvf BSR_bsds500.tgz 
fi

if [ ! -d train ]
then
	mkdir train
	mkdir train/input
	mkdir train/ground_truth
	cp BSR/BSDS500/data/images/train/*.jpg train/ground_truth/
	cp BSR/BSDS500/data/images/test/*.jpg  train/ground_truth/
	mogrify -rotate "<90" train/ground_truth/*.jpg
	#mogrify -grayscale Rec709Luma train/ground_truth/*.jpg
	images=`ls train/ground_truth`
	let i=0
	for image in ${images}
	do
		mv train/ground_truth/${image} train/ground_truth/${i}.jpg
		let i=i+1
	done
	cp train/ground_truth/*.jpg train/input/
	mogrify -quality 15 train/input/*.jpg
fi

if [ ! -d 'test' ]
then
	mkdir test
	mkdir test/input
	mkdir test/ground_truth
	cp BSR/BSDS500/data/images/val/*.jpg test/ground_truth/
	mogrify -rotate "<90" test/ground_truth/*.jpg
	#mogrify -grayscale Rec709Luma test/ground_truth/*.jpg
	images=`ls test/ground_truth`
	let i=0
	for image in ${images}
	do
		mv test/ground_truth/${image} test/ground_truth/${i}.jpg
		let i=i+1
	done
	cp test/ground_truth/*.jpg test/input/
	mogrify -quality 15 test/input/*.jpg
fi

