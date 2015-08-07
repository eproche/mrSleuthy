import sys
import click
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from gensim.models import word2vec as w2v
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import cPickle as pickle
import csv
import pandas as pd 
import py
import simplejson as json
import base64
from sleuth_out import *
from sleuth_read import *
from sleuth_crunch import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

#If there's a different set of pre-trained entity vectors that you would like to use then load them as "model"
#Comment out the line below if you've already run word2vec for a specific input, the results should be pickled.


class Config(dict):  
	"""Used to pass variables and data between commands"""
	def __init__(self, *args, **kwargs):
	    self.config = py.path.local(
	        click.get_app_dir('sleuthing')
	    ).join('config.json') # A

	    super(Config, self).__init__(*args, **kwargs)

	def load(self):
	    """load a JSON config file from disk"""
	    try:
	        self.update(json.loads(self.config.read())) # B
	    except py.error.ENOENT:
	        pass

	def save(self):
	    self.config.ensure()
	    with self.config.open('w') as f: # B
	        f.write(json.dumps(self))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        """ if input object is a ndarray it will be converted into a dict holding dtype, shape and the data base64 encoded"""
        if isinstance(obj, np.ndarray):
            data_b64 = base64.b64encode(obj.data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)


def json_numpy(dct):
    """
    Decodes a previously encoded numpy ndarray with proper shape and dtype
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

pass_config = click.make_pass_decorator(Config, ensure=True)

@click.group()
@pass_config
def cli(config):
	"""To start Mr.Sleuthy, run 'start readin INPUT', where input is the filepath you want to read inAfter 'start' has run successfully, run 'word2vec' or 'tfidf' to compute a similarity matrix
	'Output' is used after 'tfidf' or 'word2vec' to generate visual outputsTo view options for commands, enter 'start command --help' """
	config.load()
	config.save()

@cli.command('readin', short_help= click.style('Read an input file', fg='blue'))
@click.option('-t', is_flag=True, default = False, help = click.style("Use a folder of .txt files as the input", fg='cyan'))
@click.option('--des', is_flag=True, default = True, help = click.style("Only use descriptions responses", fg='blue'))
@click.option('--rem', is_flag=True, default = True, help = click.style("Only use reminded responses", fg='cyan'))
@click.option('--stem', is_flag=True, default=False, help = click.style('Only use with tfidf command! Stem words with NLTK SnowballStemmer', fg='blue'))
@click.argument('inp', default='', required = True)
@pass_config
def readin(config, inp, des, rem, t, stem):
	"""Optional flags can be used to alter how input is read
	--stem is intended to be combined with the 'tfidf' command."""
	if t:
		output = read_folder('csv/'+inp) 
	else:
		output = read_csv('csv/'+inp) 
	images = output[0]
	answers= output[1]
	config['answers'] = answers
	config['images'] = images		
	if stem:
		config['storage'] = read_words(answers, rem, des, True)
	else: 
		config['storage'] = read_words(answers, rem, des, False)
	config['inp'] = inp
	config.save()
	click.secho("\ninput has been read, ready for analysis", fg='green')
	click.secho("run tfidf or word2vec to generate similarity matrix", fg='blue', blink=False)
	pik = inp[:-4]+'_read'
	with open('pickles/'+pik, 'w') as pickle_handle:
		pickle.dump(config['storage'], pickle_handle)

@cli.command('write', short_help = click.style('Write responses, usable words, and POS counts', fg='cyan'))
@pass_config
def write(config):
	"""Write responses, usable words, and POS counts after readin"""
	clean_write(config.get('answers'))

@cli.command('tfidf', short_help = click.style('Calculate similarites with tf-idf', fg='green'))
@click.option('--pic', default='', help= click.style("load a previously read input that's been pickled", fg='green'))
@pass_config
def tfidf(config,pic):
	"""Welcome to the tfidf zone (fast)"""
	if pic != '': 
		click.secho("\ngrabbing "+pic+" from the pickle jar", fg='green')
		with open('pickles/'+pic) as pickle_handle:
			storage = pickle.load(pickle_handle)
		pic = pic[:-5]
	if pic == '':
		storage = config.get('storage')
		pic = config.get('inp')
		pic = pic[:-4]
	tf = TfidfVectorizer()
	tf_mat = tf.fit_transform(storage)
	click.secho('\ntfidf matrix generated\n', fg='green')
	sim_mat = cosine_similarity(tf_mat, tf_mat)
	json_mat = json.dumps(sim_mat, cls=NumpyEncoder)
	config['sim_mat'] = json_mat
	config.save()
	pic += '_tfidf'
	with open('pickles/'+pic, 'w') as pickle_handle:
		pickle.dump(sim_mat, pickle_handle)
	click.secho('saved tfidf sim_mat', fg='green')
	click.secho('Now run output to generate visualizations!', fg='red', blink=False)
		
@cli.command('word2vec', short_help = click.style("Calculate similarites with word2vec model", fg='yellow'))
@click.option('--unweighted', is_flag=True, default =False, help = click.style('Use with word2vec if you do not want tf-idf weightings attached', fg='yellow'))
@click.option('--pic', default='', help= click.style("load a previously read input that's been pickled", fg='yellow'))
@pass_config
def word2vec(config, unweighted, pic):
	"""Welcome to the word2vec zone (slow) Run word2vec after a file has been read in"""
	model = w2v.Word2Vec.load('/Users/epr397/Documents/googlenews_model', mmap='r') 
	model.syn0norm = model.syn0

	if pic != '': 
		click.secho("\ngrabbing "+pic+" from the pickle jar", fg='green')
		with open('pickles/'+pic) as pickle_handle:
			storage = pickle.load(pickle_handle) #load sim_mat if it's already been generated
	if pic == '':
		storage = confgi.get('storage')
	if unweighted == False: # use tf-idf weights
		tf = TfidfVectorizer()
		tf_mat = tf.fit_transform(storage) #transform documents into tf-idf matrix
		click.secho('\ntfidf matrix generated\n',fg='green')
		vocab = vocab_build(storage) #set of unique words from documents
		features = tf.get_feature_names() #the features (words) that tfidf vectorizer extracted
		features = set(features)
		vocab = vocab - (vocab - features) #get rid of discrepancies
		if vocab == features:
			click.secho("Vocab matches tfidf feature array\n", fg='green')
		vocab = sorted(list(vocab))
		tf_arr = tf_mat.toarray() 
		storage_weighted = attach_tfidf_weights(storage, vocab, tf_arr) #attaches tf-idf weighting to each words
		#calculates vector representation of document as weighted average of word2vec vectors * tf-idf weights
		feature_vecs = featureVec(storage_weighted, model, 300)  
		sim_mat = compare(storage, feature_vecs) # get cosine_similarity between each image pairing
		pic = pic[:-5]
		pic += '_w2v'
		with open('pickles/'+pic, 'w') as pickle_handle: #save the sim_mat in pickles/pickle_INPUT
			pickle.dump(sim_mat, pickle_handle)
	else: #just use word2vec
		feature_vecs_un = featureVec_unweighted(storage, model, 300)
		sim_mat = compare(storage, feature_vecs_un)
		pic += pic[:-5]
		pic += '_w2v_un'
		with open('pickles/'+pic, 'w') as pickle_handle:
			pickle.dump(sim_mat, pickle_handle)
	json_mat = json.dumps(sim_mat, cls=NumpyEncoder)
	config['sim_mat'] = json_mat #save sim_mat for outputs
	config.save()
	click.secho('saved word2vec sim_mat', fg='green')
	click.secho('Now run output to generate visualizations!', fg='red', blink=False)

@cli.command('output', short_help= click.style('Generate visual output', fg='red'))
@click.option('--no_thumb', is_flag=True, default = True, help = click.style("Don't draw the image thumbnails on iden_mat or con_mat(if images are unavailable)", fg='magenta'))
@click.option('--iden', is_flag=True, default = False, help = click.style("Generate an identity matrix with ski-kit learn manifold", fg='red'))
@click.option('--con', is_flag=True, default = False, help = click.style("Generate a confusion matrixMust include a --sep='index' flag", fg='magenta'))
@click.option('--sep', default = 1, help= click.style("Required for --con! sep=index separating two categories in the document/image ordering.", fg='red'))
@click.option('--spring', is_flag=True, default = False, help = click.style("Generate a graph in spring layout with pyplot", fg='magenta'))
@click.option('--mds', is_flag=True, default = False, help = click.style("Generate an MDS plot", fg='red'))
@click.option('--im_mds', is_flag=True, default = False, help = click.style("Generate an MDS plot with images thumbnails", fg='red'))
@click.option('--vis', is_flag=True, default= False, help = click.style('Write results to "nodes.txt" and "edges.txt" in vis.js graph formatThen you can manually copy into graph_vis.html', fg='magenta'))
@click.option('--explore', is_flag=True, default=False, help= click.style('Explore how the spring graph changes over a range of thresholds, specified by --thresh1=X and --thresh2=Y', fg='red'))
@click.option('--thresh1', default=0.6, help= click.style("Threshold to use for spring graphDefault = 0.6Also sets start threshold for --explore option", fg='magenta'))
@click.option('--thresh2', default=0.7, help= click.style("Threshold to use for vis.js outputDefault = 0.7Also sets the end threshold for --explore option", fg='red'))
@click.option('--step', default=0.01, help= click.style("Step size for explore optionDefault = 0.1", fg='magenta'))
@click.option('--pic', default='', help= click.style("load a previously calculated input that's been pickled", fg='red'))
@pass_config
def output(config, spring, mds, im_mds, iden, con, sep, vis, no_thumb, explore, thresh1, thresh2, step, pic):
	"""Generate visual outputs after running tfidf or word2vec"""
	if pic != '':
		click.secho("\ngrabbing "+pic+" from the pickle jar", fg='green')
		with open('pickles/'+pic) as pickle_handle:
			results = pickle.load(pickle_handle)
		inp = 'csv/'+pic[:-4]+'.csv'
		output = read_csv(inp) 
		images = output[0]
	if pic == '':	
		json_mat = config.get('sim_mat')
		results = json.loads(json_mat, object_hook=json_numpy)
		images = config.get('images')
	print results
	if iden:
		click.secho('\ngenerating identity matrix output at iden_mat.png\n', fg='red', blink=False)
		clus = cluster(results, config.get('images'), False) #hierarchal centroid clustering
		clusmat = clus[0]
		clusim = clus[1] #new image ordering
		draw_id(clusmat, clusim, no_thumb) 
	if con:
		slices = conslice(results, sep) 
		clus = cluster(slices, config.get('images'), True)
		clusmat = clus[0]
		clusim = clus[1]
		click.secho('generating confusion matrix output at con_mat.png\n', fg='blue', blink=False)
		draw_con(clusmat, config.get('images'), clusim, sep)
	if spring: 
		click.secho('displaying spring graph\n', fg='yellow', blink=False)
		draw_spring_graph(results, thresh1, sep)
	if mds:
		click.secho('displaying MDS plot\n', fg='magenta', blink=False)
		mds_draw(results, sep)
	if im_mds:
		click.secho('displaying MDS plot with image thumbnails\n', fg='magenta', blink=False)
		draw_mds_img(results, sep, images)
	if vis:
		res_vis(results, images, thresh2)
		click.secho('vis.js formatted graph sent to nodes.txt and edges.txt\n', fg='green')
	if explore:
		explore(results, thresh1, thresh2, step)
		click.secho('Saving exploration to explore_results', fg='green', blink=False)










