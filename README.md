# Mr. Sleuthy
Calculates similarity measures of image descriptions and produces visualizations

<hr />
This project uses a virtualenv to load requirements. To generate the virtualenv run:

```bash
virtualenv venv
```

in the mrSleuthy directory. You can then activate the virtural environment and load mrSleuthy by running:

```bash
. venv/bin/activate
pip install --editable .
```

If everything loads properly, then running:

```bash
start
```

from the terminal should display the available options and commands, as shown below.

```bash 
start
Usage: start [OPTIONS] INP COMMAND [ARGS]...

  To run Mr. Sleuthy, enter commands as follows: start "[options] "input
  filename", [command]"  Make sure to include an input flag that corresponds
  with the input file you want to use.

Options:
  -m             Use an unedited MTurk batch.csv file as the input. Can be
                 used with --des and --rem flags
  -t             Use a folder of .txt files as the input
  -c             Use a two-column csv file as input
  --spring       Generate a graph in spring layout with pyplot
  --iden         Generate an identity matrix with ski-kit learn manifold
  --mds          Generate an MDS plot
  --con          Generate a confusion matrix. Must include a --sep='index'
                 flag
  --sep INTEGER  Required for --con! sep=index separating two categories in
                 the document/image ordering.
  --vis          Write results to "nodes.txt" and "edges.txt" in vis.js graph
                 format. Then you can manually copy into graph_vis.html
  --des          Only use descriptions responses
  --rem          Only use reminded responses
  --no_thumb     Do not draw the image thumbnails on iden_mat or con_mat. (if
                 images are unavailable)
  --stem         Only use with tfidf command! Stem words with NLTK
                 SnowballStemmer
  --unweighted   Use with word2vec if you do not want tf-idf weightings
                 attached
  --help         Show this message and exit.

Commands:
  tfidf     Welcome to the tfidf zone (fast)
  word2vec  Welcome to the word2vec zone (slow)
  write     Write responses, usable words, and POS counts
(venv)rowlfthedog:mrSleuthy epr397$ 

```

<hr />

## Contents 
The csv folder contains sample inputs from lewpealab MTurk studies and the Flikr8k dataset, which includes over 8000 Flikr images with 5 single-sentence, MTurk-generated descriptions per image.
Read about it [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)

M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artificial Intelligence Research, Volume 47, pages 853-899

Within the examples folder, each folder title is a sample command, and the contents of each folder are what that command outputs

##Usage
The tfidf command and the write command will work without any additional files. To use the word2vec command, you need to load pre-trained vectors. You can set the path in line 25 of sleuthin.py. The default model uses the Google News dataset (1.65Gb), which can be obtained at the [word2vec site](https://code.google.com/p/word2vec/)

Loading a large model into memory can take quite a bit of time. Once you run a specific input file through word2vec, the results are pickled. If you want to rerun the same input with different visual outputs, comment out line 25. Sleuthin.py will load the pickled results. 

You can make some changes to the Google News model that will allow it load faster on machines with less memory. Once you've activated the virtual environment and downloaded the Google News dataset from [this Google Drive folder](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit), open an interactive Python shell by running:
```bash
python
```
Then, in python, run the following commands
```python
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
#load the original Google News model
model = word2vec.Word2Vec.load_word2vec_format('path to folder where you downloaded/GoogleNews-vectors-negative300.bin.gz', binary=True)
#you should see the following outputs(with different times)
2015-07-07 13:41:35,279 : INFO : loading projection weights from your path/GoogleNews-vectors-negative300.bin
2015-07-07 13:42:44,557 : INFO : loaded (3000000, 300) matrix from your path/GoogleNews-vectors-negative300.bin
2015-07-07 13:42:44,557 : INFO : precomputing L2-norms of word weight vectors
```
Now, to indicate that the model is no longer being trained, run:
```python
model.init_sims(replace=True)
model.save('path to wherever you want the new model saved/googlenews_model')
#you should see
2015-07-07 13:46:52,846 : INFO : saving Word2Vec object under googlenews_model, separately None
2015-07-07 13:46:52,846 : INFO : not storing attribute syn0norm
2015-07-07 13:46:52,846 : INFO : storing numpy array 'syn0' to googlenews_model.syn0.npy
```
Finally, in line 25 of sleuthin.py, set the filepath to wherever your 'googlenew_model' is stored. The model should now load noticably faster.



