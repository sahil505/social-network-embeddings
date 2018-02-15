#!/bin/bash
indir="sentences_files"
outfile="sentences_files/userSentencesComb"
for string in "$indir"/*.txt ; do
	cat "$string" >> "$outfile"
	echo "$string"
	# rm "$string"
done