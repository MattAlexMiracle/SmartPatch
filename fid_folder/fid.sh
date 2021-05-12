echo "Baseline"
python fid_score_crop64x64.py --gpu 0 --batch-size=256 ../Benchmark/research-seq2seq-HTR/dataset/full/BaselineDataset ../dataset
echo "Naive"
python fid_score_crop64x64.py --gpu 0 --batch-size=256 ../Benchmark/research-seq2seq-HTR/dataset/full/NaiveDataset ../dataset
echo "Smart"
python fid_score_crop64x64.py --gpu 0 --batch-size=256 ../Benchmark/research-seq2seq-HTR/dataset/full/SmartDataset ../dataset
echo "Character"
python fid_score_crop64x64.py --gpu 0 --batch-size=256 ../Benchmark/research-seq2seq-HTR/dataset/full/CharacterDataset ../dataset
#python fid_score_crop64x64.py --gpu 0 --batch-size=256 res_2.in_vocab_te_writer ../datasets
#python fid_score_crop64x64.py --gpu 0 --batch-size=256 res_3.oo_vocab_tr_writer ../datasets
#python fid_score_crop64x64.py --gpu 0 --batch-size=256 res_4.oo_vocab_te_writer ../datasets
