import matplotlib.pyplot as plt
import click
import sys, os
from sklearn import manifold
import numpy as np 
from numpy import *
from skimage import io, img_as_float, color
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import networkx as nx
import scipy  
import pandas as pd 
from sklearn import manifold
import scipy.cluster.hierarchy as hc 
import scipy.cluster.hierarchy as hc 
from PIL import Image, ImageDraw, ImageColor
from PIL import Image
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, \
     AnnotationBbox
import shutil
import webbrowser

def draw_id(results, stored_images, thumb):
	"""draws an indentity matrix from the similarity matrix"""
	stored = []
	for filename in stored_images:
		stored.append("mturk_images/"+filename) #edit to match unique image folder name
	stored_images1 = stored
	stored_images2 = stored[::-1] #reverse for x-axis thumbs
	im = Image.new('RGB', ((len(stored_images2)+2)*50, (len(stored_images1)+2)*50), (255, 255, 255))
	draw = ImageDraw.Draw(im)
	if thumb == True:
		click.secho('drawing thumbnails', fg='red', blink=False)
		for i in range(len(stored_images2)):
			sys.stdout.write(".")
			sys.stdout.flush()
			thumb = Image.open(stored_images2[i]) #stored_images_for_ident
			thumb = thumb.resize((50, 50), Image.ANTIALIAS)
			im.paste(thumb, box=(50*i+50, 0, 50*i+100, 50)) #box position =(x1, y1, x2, y2) from top left
			im.paste(thumb, box=(50*i+50, 50*len(stored_images2)+50, 50*i+100, 50*len(stored_images2)+100)) 
			#draw on both sides for ease of viewing
		sys.stdout.write('\n')
		sys.stdout.flush()
		for j in range(len(stored_images1)):
			sys.stdout.write(".")
			sys.stdout.flush()
			thumb = Image.open(stored_images1[j])
			thumb = thumb.resize((50, 50), Image.ANTIALIAS)
			im.paste(thumb, box=(0, 50*j+50, 50, 50*j+100)) 
			im.paste(thumb, box=(50*len(stored_images1)+50, 50*j+50, 50*len(stored_images1)+100, 50*j+100))
		sys.stdout.write('\n')
		sys.stdout.flush()	
	# Draw color squares
	for j in range(len(results)):
		for k in range(len(results)):
			avg_score = results[j][k]
			r_amount = 0
			g_amount = 0
			b_amount = 0
			# if avg_score >= 0.6:
			# 	r_amount = 255
			# 	g_amount = 255 - 255*4*(avg_score-0.6)
			# if avg_score >= 0.4 and avg_score <= 0.6:
			# 	r_amount = 255*5*(avg_score-0.4)
			# 	g_amount = 255
			# if avg_score >= 0.2 and avg_score <= 0.4:
			# 	g_amount = 255
			# 	b_amount = 255 - 255*5*(avg_score-0.2)
			# if avg_score <= 0.2:
			# 	g_amount = 255*4*(avg_score)
			# 	b_amount = 255
			# if avg_score >= 0.51:
			# 	r_amount = 255
			# 	g_amount = 255 - 255*4*(avg_score-0.51)
			# if avg_score >= 0.34 and avg_score <= 0.51:
			# 	r_amount = 255*5*(avg_score-0.34)
			# 	g_amount = 255
			# if avg_score >= 0.17 and avg_score <= 0.34:
			# 	g_amount = 255
			# 	b_amount = 255 - 255*5*(avg_score-0.17)
			# if avg_score <= 0.17:
			# 	g_amount = 255*4*(avg_score)
			# 	b_amount = 255
			if avg_score >= 0.75: # yellow to red
				r_amount = 255
				g_amount = 255 - 255*4*(avg_score-0.75)
			if avg_score >= 0.5 and avg_score <= 0.75: #green to yellow
				r_amount = 255*4*(avg_score-0.5)
				g_amount = 255
			if avg_score >= 0.25 and avg_score <= 0.5: #cyan to green
				g_amount = 255
				b_amount = 255 - 255*4*(avg_score-0.25)
			if avg_score <= 0.25: #blue to cyan
				g_amount = 255*4*(avg_score)
				b_amount = 255
			score = round(avg_score, 3)
			score = str(score)
			draw.rectangle([k*50+50, j*50+50, k*50+100, j*50+100], fill=(int(r_amount), int(g_amount), int(b_amount)), outline=None)
			draw.text((k*50+58, j*50+80), score, fill=(0,0,0))
	del draw
	im.save('iden_mat.png', "PNG")

def draw_con(results, stored_images, clusim, sep):
	"""draw confusion matrix separating two categories, determined by sep"""
	#normalize confusion matrix
	images = clusim 
	scenes = stored_images[sep:]
	conmax = np.amax(results) #normalization setup
	conmin = np.amin(results)
	conmax = conmax - conmin
	thumbs_x = [""]*len(scenes)
	for i in range(len(scenes)):
		thumbs_x[i] = "mturk_images/"+scenes[i]
	thumbs_y = [""]*len(images)
	for j in range(len(images)):
		thumbs_y[j] = "mturk_images/"+images[j]
	dim = shape(results) 
	x = dim[1]
	y = dim[0]
	im = Image.new('RGB', ((x+2)*50, (y+2)*50), (255, 255, 255))
	draw = ImageDraw.Draw(im)
	click.secho('drawing thumbnails', fg='blue', blink=False)
	for j in range(y):
		sys.stdout.write(".")
		sys.stdout.flush()
		thumb = Image.open(thumbs_y[j])
		thumb = thumb.resize((50, 50), Image.ANTIALIAS)
		im.paste(thumb, box=(0, 50*j+50, 50, 50*j+100))
		im.paste(thumb, box=(50*x+50, 50*j+50, 50*x+100, 50*j+100))
	sys.stdout.write('\n')
	sys.stdout.flush()
	for i in range(x):
		sys.stdout.write(".")
		sys.stdout.flush()
		thumb = Image.open(thumbs_x[i]) #images_for_ident
		thumb = thumb.resize((50, 50), Image.ANTIALIAS)
		im.paste(thumb, box=(50*i+50, 0, 50*i+100, 50))
		im.paste(thumb, box=(50*i+50, 50*y+50, 50*i+100, 50*y+100))
	sys.stdout.write('\n')
	sys.stdout.flush()
	#Draw color squares
	#Images and scenes tend to have lower similarities
	#Colors are adjusted to (0, .2, .4, .6) scale to compensate
	for j in range(y):
		for k in range(x):
			avg_score = results[j][k]
			avg_score = (avg_score - conmin) / conmax
			r_amount = 0
			g_amount = 0
			b_amount = 0
			if avg_score >= 0.6:
				r_amount = 255
				g_amount = 255 - 255*4*(avg_score-0.6)
			if avg_score >= 0.4 and avg_score <= 0.6:
				r_amount = 255*5*(avg_score-0.4)
				g_amount = 255
			if avg_score >= 0.2 and avg_score <= 0.4:
				g_amount = 255
				b_amount = 255 - 255*5*(avg_score-0.2)
			if avg_score <= 0.2:
				g_amount = 255*4*(avg_score)
				b_amount = 255
			score = round(avg_score, 3)
			score = str(score)
			draw.rectangle([k*50+50, j*50+50, k*50+100, j*50+100], fill=(int(r_amount), int(g_amount), int(b_amount)), outline=None)
			draw.text((k*50+58, j*50+80), score, fill=(0,0,0))
	del draw
	im.save('con_mat.png')

def draw_spring_graph(results, thresh, sep):
	"""Draw graph in spring layoout, 
	force-directed algorithm puts similar image nodes close to eachother
	Assumes symmetric split with the two categories (artificial/natural + indoor/outdoor for ComCon)"""
	img_colors = ['green'] * (sep/2) #add or modify indices to get new colors 
	img_colors2 = ['red'] * (sep/2)
	sce_colors = ['blue'] * ((len(results) - sep)/2)
	sce_colors2 = ['yellow'] * ((len(results) - sep)/2)
	node_colors = img_colors + img_colors2 + sce_colors +sce_colors2
	res2 = np.copy(results)
	low_val = res2 < thresh
	res2[low_val] = 0
	graph = nx.from_numpy_matrix(res2)
	pos = nx.spring_layout(graph)
	nx.draw_networkx_nodes(graph, pos=pos, node_color = node_colors)
	nx.draw_networkx_edges(graph, pos=pos)
	# xs = [] # Add labels (looks pretty messy with large graph)
	# ys = []
	# for i in range(len(pos)):
	# 	xs.append(pos[i][0])
	# 	ys.append(pos[i][1])
	# for label, x, y, in zip(names, xs, ys):
	# 	plt.annotate(
	# 	label,
	# 	xy = (x, y), xytext = (-10, 35),
	# 	textcoords = 'offset points', ha = 'right', va = 'bottom',
	# 	bbox = dict(boxstyle = 'round,pad=0.5', fc = 'green', alpha = 0.7),
		# arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
	plt.show()


def mds_draw(results, sep):
	"""draws MDS plot of high-dimension vectors"""
	mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
	a = (sep/2) #modify indices to match data
	b = sep
	c = ((len(results)-sep)/2)+b
	# c = 227
	res2 = np.copy(results)
	res2 = 1 - results
	result = mds.fit(res2)
	coords = result.embedding_
	plt.subplots_adjust(bottom = 0.1)
	plt.scatter(
		coords[:a, 0], coords[:a, 1], color = 'green', marker = 'o', s=70)
	plt.scatter(
		coords[a:b, 0], coords[a:b, 1], color = 'red', marker = 'o', s=70)
	plt.scatter(
		coords[b:c, 0], coords[b:c, 1], color = 'blue', marker = 'v', s=70)
	plt.scatter(
		coords[c:, 0], coords[c:, 1], color = 'orange', marker = 'v', s=70)
	# for label, x, y in zip(names, coords[:, 0], coords[:, 1]):
	#     plt.annotate(
	#         label,
	#         xy = (x, y), xytext = (0,0),
	#         textcoords = 'offset points', ha = 'right', va = 'bottom',
	#         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'green', alpha = 1.0),
	#         )
	plt.draw()
	plt.show()

def draw_mds_img(results, sep, images):
	stored = []
	for filename in images:
	    stored.append("mturk_images/"+filename)
	mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
	a = (sep/2) #modify indices to match data
	b = sep
	c = ((len(results)-sep)/2)+b
	# c = 227
	res2 = np.copy(results)
	res2 = 1 - results
	result = mds.fit(res2)
	coords = result.embedding_
	coor = coords

	corners = []
	for i in coords:
		corners.append((i[0],i[1],i[0]+50,i[1]+50))
	corners = corners[:10]
	thumbs = []
	for i in range(len(images)):
		if i<sep:
			image = io.imread(stored[i])
			white = np.array([0,0,0,0])
			mask = np.abs(image - white).sum(axis=2) < 0.05
			coords = np.array(np.nonzero(~mask))
			top_left = np.min(coords, axis=1)
			bottom_right = np.max(coords, axis=1)
			out = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
			thumb = Image.fromarray(np.uint8(out))
			thumb = thumb.resize((50, 50), Image.ANTIALIAS)
			thumbs.append(thumb)
		else:
			image = io.imread(stored[i])
			image2 = np.copy(image)
			image3 = np.copy(image)
			image4 = np.dstack((image,image2,image3))
			print image4.shape
			thumb=Image.fromarray(np.uint8(image4))
			thumb = thumb.resize((50, 50), Image.ANTIALIAS)
			thumbs.append(thumb)
		print i

	fig, ax = plt.subplots()
	for xy, img in zip(coor, thumbs):
		imbox = OffsetImage(img, zoom=0.6)
		ab = AnnotationBbox(imbox, xy, frameon=False)
		ax.add_artist(ab)

	ax.set_xlim(-1,1)
	ax.set_ylim(-1,1)
	plt.draw()
	plt.show()


def res_vis(results, images, thresh):
	"""Saves results in vis.js formatted .txt files. 
	After generating, manually copy nodes.txt and edges.txt into graph_vis.html template,
	then delete .txt files
	"""
	res2 = np.copy(results)
	nodes = []
	edges = []
	connect = []
	for i in range(len(results)):
		for j in range(i+1, len(results)):
			if res2[i][j] >= thresh:
				string = "{from: "+str(i)+", to: "+str(j)+"},"
				string2 = "{id: " +str(i)+ ", image: 'mturk_images/" +images[i]+ "'},"
				edges.append(string)
				connect.append(i)

	for i in range(len(results)):
		if i in connect:
			string = "{id: " +str(i)+ ", image: 'mturk_images/" +images[i]+ "'},"
			nodes.append(string)	
	f=open('graph_vis.html', 'r')
	shutil.copy2('graph_vis.html','graph_vis_out.html')
	contents = f.readlines()
	f.close()
	for i in range(len(nodes)):
		contents.insert(21+i,nodes[i]+'\n')
	start = len(nodes)+26
	for i in range(len(edges)):
		contents.insert(start+i,edges[i]+'\n')
	f=open("graph_vis_out.html", 'w')
	contents = "".join(contents)
	f.write(contents)
	f.close()
	new = 2
	webbrowser.open('file://'+os.path.realpath('graph_vis_out.html'),new=new)


def explorate(results, thresh1, thresh2, step, sep):
	"""view spring graphs and degree distributions across a range of thresholds
	Default is plt.show() for each step, uncomment last lines to save figures in explore_results folder
	"""
	i = 0
	img_colors = ['green'] * (sep/2) #indices separating different node colors
	img_colors2 = ['red'] * (sep/2)
	sce_colors = ['blue'] * ((len(results) - sep)/2)
	sce_colors2 = ['yellow'] * ((len(results) - sep)/2)
	node_colors = img_colors + img_colors2 + sce_colors +sce_colors2
	for thresh in np.arange(thresh1, thresh2, step):
		print thresh
		res2 = np.copy(results)
		hi_val = (res2 >= thresh)
		graph = nx.from_numpy_matrix(hi_val)
		pos = nx.spring_layout(graph)
		nx.draw_networkx_nodes(graph, pos=pos, node_color = node_colors)
		nx.draw_networkx_edges(graph, pos=pos)
 		plt.show()
 		# plt.savefig('explore_results/spring'+str(i)+'.png')
 		#plt.close('all')
 		i+=1

 	i = 0
	for thresh in np.arange(thresh1, thresh2, step):
		print thresh
		res2 = np.copy(results)
		hi_val = (res2 >= thresh)
		graph = nx.from_numpy_matrix(hi_val)		
		degrees = graph.degree()
		values = sorted(set(degrees.values()))
		hist = [degrees.values().count(x) for x in values]

		plt.figure()
		plt.grid(True)
		plt.plot(values, hist, 'ro-') 
		plt.legend(['degree'])
		plt.xlabel('Degree')
		plt.ylabel('Number of nodes')
		plt.title('MTurk Degree Distribution')
		plt.xlim([0, 35])
		plt.ylim([0, 25])
		plt.show()
		# plt.savefig('explore_results/distribution'+str(i)+'.png')
		# plt.close('all')
		i += 1

def conslice(sim_mat, sep):
	"""Slices a confusion matrix out of similarity matrix based on sep"""
	images = sim_mat[:sep]
	slices = []
	for i in range(len(images)):
		slices.append(images[i][sep:])
	return slices 

def cluster(sim_mat, images, con):
	"""hierachal clustering"""
	if con: #only cluster along y-axis
		dim = shape(sim_mat)
		x = dim[0]
		images = images[:x]
	df = pd.DataFrame(sim_mat)
	hier = hc.linkage(sim_mat, method='centroid')
	lev = hc.leaves_list(hier)
	# get the clustered indices
	mat = df.iloc[lev,:]
	if not con:
		mat = mat.iloc[:, lev[::-1]]
	#get hierarchal indices back to numpy matrix
	simul = mat.as_matrix()
	#to get new image list assuming original list is ims
	imin = mat.index.values
	clusim = [""]*len(images)
	for i in range(len(images)):
		clusim[i] = images[imin[i]]
	return [simul, clusim]