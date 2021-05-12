python generatePseudoIAM.py save_weights_naive Benchmark/research-seq2seq-HTR/dataset/NaiveDataset --patch_loss
python generatePseudoIAM.py save_weights_baseline Benchmark/research-seq2seq-HTR/dataset/BaselineDataset
python generatePseudoIAM.py save_weights_smart Benchmark/research-seq2seq-HTR/dataset/SmartDataset --smart_patch_loss
python generatePseudoIAM.py save_weights_character Benchmark/research-seq2seq-HTR/dataset/CharacterDataset --character_patch_loss
