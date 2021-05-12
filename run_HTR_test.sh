cd Benchmark/research-seq2seq-HTR/
echo "baseline"
python test.py 126 HTR_REAL_weights BaselineDataset
echo "naive patch"
python test.py 126 HTR_REAL_weights NaiveDataset
echo "smart patch"
python test.py 126 HTR_REAL_weights SmartDataset
echo "character patch"
python test.py 126 HTR_REAL_weights CharacterDataset
