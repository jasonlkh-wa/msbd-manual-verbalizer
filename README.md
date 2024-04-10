# How to run the model.py
`python model.py ...`
The available flags are
- dataset: the name of the dataset, this is defined in the code to map to the config (e.g. prompt, dataset)
- k-shot: number of shot for training
- epoch

# Where is the output result
All results are stored in `output/` with the file naming convention of 
[dataset]_[kshot]_[epoch].parquet - for label output (text included)
[dataset]_[kshot]_[epoch]_accuracy.txt - just a plain text storing the test accuracy

# How to setup the dataset
Under `data/[dataset name]`
create 1 [*parquet] with shape (x, 2), 1st col will be the text and 2nd col will be  the integer label
create 1 [classes.txt] where each row is the name of the class
[train.parquet] and [validation.parquet] are not required as the train-test split is performed in runtime, so 1 single data file is good.

Next, edit [read_data], [load_class_map], [get_manual_template], [get_verbalizer] to setup the per-dataset configs