mkdir automl_inputs
cd automl_inputs
mkdir 0 1 2
grep '\tnegative' ../train_simple.tsv | cut -f 1 > 0/negatives.txt
grep '\tneutral' ../train_simple.tsv | cut -f 1 > 1/neutrals.txt
grep '\tpositive' ../train_simple.tsv | cut -f 1 > 2/positives.txt
cd 0
# we use the high suffix length of 3 allowing up to 17,576 files
split -a 3 -l 10 negatives.txt
for f in x*; do mv "$f" "$f.txt"; done
rm negatives.txt
cd ../1
split -a 3 -l 10 neutrals.txt
for f in x*; do mv "$f" "$f.txt"; done
rm neutrals.txt
cd ../2/
split -a 3 -l 10 positives.txt
for f in x*; do mv "$f" "$f.txt"; done
rm positives.txt
cd ../..
zip -r automl_inputs.zip automl_inputs

