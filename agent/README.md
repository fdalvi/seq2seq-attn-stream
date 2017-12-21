# Simultaneous decoding agent

This is a generic agent that learns when to segment a source sentence from a dataset. The dataset is of the form:

| Input  | Label |
| ------------- | ------------- |
| `<S> <S> <S> <S> Hello` | `Segment`  |
| `<S> <S> <S> Hello How` | `Don't Segment`  |
| `<S> <S> Hello How are` | `Don't Segment`  |
| `<S> Hello How are you` | `Segment`  |
| `Hello How are you today` | `Segment`  |

The number of tokens in the input is defined by the `num_feats` variable. In the above example, it is `5`. The input files must have `num_feats+1` tab-separated columns with each instance on its own line. The last column must define the label.

#### Preprocessing
The data must be first preprocessed using the `preprocess.lua` script. The options are as follows:
* `-input`: Path to the input file as defined above
* `-output`: Path to the folder where the processed data will be saved. Multiple files will be saves (vocabs, training data, validation data etc.)
* `-num_feats`: Number of input tokens per instance
* `-shuffle`: Boolean defining if the data should be shuffled before saving.

#### Training
Once the files have been preprocessed, the `train.lua` script can be used to train an agent. The options are as follows:
* `-input`: Path to the processed data directory
* `-output`: Path to the folder where the model checkpoints will be saved
* `-use_gpu`: Whether or not the computation should be done on a GPU
* `-many_hot`: Whether a `many_hot` representation should be used instead of `one_hot` for the input representation
