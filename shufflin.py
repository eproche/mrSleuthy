
import numpy as np
from numpy import *
import cPickle as pickle
from scipy import sparse
import csv
from PIL import Image, ImageDraw, ImageColor

def read_csv(path):
	answers = []
	images = []
	with open(path, 'rU') as csvfile:
		reader = csv.reader(csvfile)
		index = 0
		for row in reader:
			im = (row[0])
			bothanswers = (row[1])
			split = bothanswers.split('|')
			if im not in images:
				images.append(im)
				answers.append([[split[0]],[split[1]]])
				index += 1
			else:
				answers[index-1][0].append(split[0])
				answers[index-1][1].append(split[1])
	out = []
	out.append(images)
	out.append(answers)
	return out 

def conslice(sim_mat, sep):
	images = sim_mat[:sep] 
	slices = []
	for i in range(len(images)):
		slices.append(images[i][sep:])
	return slices

pik = 'pickles/pickle_study.csv'
with open(pik) as pi:
	sim_mat = pickle.load(pi)
conmat = conslice(sim_mat, 200)
conmat = np.transpose(conmat)

inds = argsort(conmat, axis=1)
uniq = inds[:,:200]
unis = unique(uniq)
print len(unis)

natmat = conmat[:,:100]
artmat = conmat[:,100:]
artind = argsort(artmat, axis=1)
natind = argsort(natmat, axis=1)
c = zeros((64,200))
artind = artind+100
c[:,1::2]=natind
c[:,::2]=artind
# conmax = np.amax(conmat)
# conmin = np.amin(conmat)
# conmax = conmax-conmin
# conmat = ((conmat-conmin)/conmax)
#@profile
def opto(conmat, thresh, withthresh):
	nat = np.arange(100)
	art = np.arange(100,200)
	allim = arange(200)
	ind = np.arange(64)
	# inds = zip(ind,nat)
	# nat2 = [conmat[x][y] for x, y in inds]
	# nat3 = zip(nat, nat2)
	# inds2 = zip(ind, art)
	# art2 = [conmat[x][y] for x, y in inds2]
	# art3 = zip(art, art3)
	count = 1
	contin = True
	f = lambda x, y : conmat[x][y]
	while contin:
		ranall = random.permutation(allim)
		arr1 = ranall[:64]
		arr2 = ranall[64:128]
		arr3 = ranall[128:192]
		# rannat = np.random.permutation(nat)
		# ranart = np.random.permutation(art)
		# nat1 = rannat[:32]
		# #nat1.shape=(4,8)
		# nat2 = rannat[32:64]
		# #nat2.shape=(2,16)
		# nat3 = rannat[64:96]
		# art1 = ranart[:32]
		# #art1.shape=(4,8)
		# art2 = ranart[32:64]
		# #art2.shape=(2,16)
		# art3 = ranart[64:96]

		# arr1=np.zeros((64), dtype=np.int)
		# arr2=np.zeros((64), dtype=np.int)
		# arr3=np.zeros((64), dtype=np.int)		

		# arr1[0::8]=nat1[0]
		# arr1[1::8]=nat1[1]
		# arr1[2::8]=nat1[2]
		# arr1[3::8]=nat1[3]

		# arr1[4::8]=art1[0]
		# arr1[5::8]=art1[1]
		# arr1[6::8]=art1[2]
		# arr1[7::8]=art1[3]

		# arr2[0::4]=nat2[0]
		# arr2[1::4]=nat2[1]

		# arr2[2::4]=art2[0]
		# arr2[3::4]=art2[1]
		# arr1[0::2]=art1
		# arr1[1::2]=nat1

		# arr2[0::2]=nat2
		# arr2[1::2]=art2

		# arr3[0::2]=nat3
		# arr3[1::2]=art3
		keepgoing = True
		for i in range(64):
			sums = sim_mat[arr1[i]][arr2[i]]+sim_mat[arr1[i]][arr3[i]]+sim_mat[arr2[i]][arr3[i]]
			within = sums/3
			if within > withthresh:
				keepgoing = False
				break
		if keepgoing:
			gets1 = zip(ind, arr1)
			gets2 = zip(ind, arr2)
			gets3 = zip(ind, arr3)
			allcord = gets1+gets2+gets3
			coords = zip(*allcord)
			mask = sparse.coo_matrix((ones(len(coords[0])), coords), shape = conmat.shape, dtype = 'bool')
			mask1 = mask.toarray()
			val = np.multiply(conmat,mask1)
			sums = np.sum(val, axis=1)
			#variances = np.var(val2, axis=1)
			checks = (sums/3.0) < thresh
			#checks2 = val < tot
			check = np.all(checks)
			#check2 = np.all(checks2)
			if check:
				arr = np.column_stack((arr1, arr2, arr3))
				vals = val[where(val>0)]
				vals2 = reshape(vals,(64,3))
				print [vals2,arr]
				return arr
				contin = False
		count += 1
		if (count%1000 == 0):
			print count

#@profile
def opto2(conmat, thresh, tot):
	nat = np.arange(100)
	art = np.arange(100,200)
	ind = np.arange(64)
	# inds = zip(ind,nat)
	# nat2 = [conmat[x][y] for x, y in inds]
	# nat3 = zip(nat, nat2)
	# inds2 = zip(ind, art)
	# art2 = [conmat[x][y] for x, y in inds2]
	# art3 = zip(art, art3)
	natmat = conmat[:,:100]
	artmat = conmat[:,100:]
	artind = argsort(artmat, axis=1)
	natind = argsort(natmat, axis=1)
	uniq = zeros((64,200))
	artind = artind+100
	uniq[:,1::2]=natind
	uniq[:,::2]=artind
	print uniq[:,:9]
	count = 1
	contin = True
	f = lambda x, y : conmat[x][y]

	while contin:
		# for i in range(len(uniq)):
		# 	firs = uniq[i][0:6]
		# 	ranfirs = random.permutation(firs)
		# 	uniq[i][0] = ranfirs[0]
		# 	uniq[i][1] = ranfirs[1]
		# 	uniq[i][2] = ranfirs[2]
		# 	uniq[i][3] = ranfirs[3]
		# 	uniq[i][4] = ranfirs[4]
		# 	uniq[i][5] = ranfirs[5]
		# rannat = np.random.permutation(nat)
		# ranart = np.random.permutation(art)
		ranind = np.random.permutation(ind)
		rans = list(ranind)
		arr = zeros((64,3))
		arr2 = zeros((64,3))
		usedinds = set()
		for i in rans:
			j = 0
			k = random.randint(2)
			go = True
			while go:
				
				if str(uniq[i][k]) in usedinds:
					k += 1
				else:
					arr[i][j] = conmat[i][uniq[i][k]]
					arr2[i][j] = uniq[i][k]
					usedinds.add(str(uniq[i][k]))
					go = False
		ranind2 = np.random.permutation(ind)
		rans2 =list(ranind2)
		for i in rans2:
			j = 1
			k = random.randint(2)
			go = True
			while go:
				if str(uniq[i][k]) in usedinds:
					k+=1
				else:
					arr[i][j] = conmat[i][uniq[i][k]]
					arr2[i][j] = uniq[i][k]
					usedinds.add(str(uniq[i][k]))
					go = False
		ranind3 = np.random.permutation(ind)
		rans3 =list(ranind3)
		for i in rans3:
			j = 2
			k = random.randint(2)
			go = True
			while go:
				if str(uniq[i][k]) in usedinds:
					k+=1
				else:
					arr[i][j] = conmat[i][uniq[i][k]]
					arr2[i][j] = uniq[i][k]
					usedinds.add(str(uniq[i][k]))
					go = False
		# print arr2
		# for i in (arr2):
		# 	print str(sim_mat[i[0]][i[1]])+": "+str(sim_mat[i[0]][i[2]])+": "+str(sim_mat[i[1]][i[2]])
		check2 = True
		withs = zeros((64))
		j=0
		for i in (arr2):
			within = (sim_mat[i[0]][i[1]]+sim_mat[i[0]][i[2]]+sim_mat[i[1]][i[2]])/3
			withs[j]=within
			j+=1
			if within > tot:
				check2 = False
				continue
		sums = np.sum(arr, axis=1)
		checks = (sums/3.0) < thresh
		#variances = np.var(arr, axis=1)
		#checks2 = variances < tot
		check = np.all(checks)
		#check2 = np.all(checks2)
		if check and check2:
			print arr
			#print arr2
			print withs
			return arr2
			
			#print variances
			contin = False
		count += 1
		if (count%100 == 0):
			print count


def draw_group(results, scenes, images):
	#normalize confusion matrix
	thumbs_x = [""]*len(scenes)
	for i in range(len(scenes)):
		thumbs_x[i] = "mturk_images/"+scenes[i]
	thumbs_y = [""]*len(images)
	for j in range(len(images)):
		thumbs_y[j] = "mturk_images/"+images[j]

	im = Image.new('RGB', ((5)*150, (65)*150), (255, 255, 255))
	draw = ImageDraw.Draw(im)

	for i in range(len(results)):
		thumb = Image.open(thumbs_x[i])
		thumb = thumb.resize((100, 100), Image.ANTIALIAS)
		im.paste(thumb, box=(50, 150*i+50, 150, 150*i+150))

		thumb = Image.open(thumbs_y[results[i][0]])
		thumb = thumb.resize((100, 100), Image.ANTIALIAS)
		im.paste(thumb, box=(200, 150*i+50, 300, 150*i+150))

		thumb = Image.open(thumbs_y[results[i][1]])
		thumb = thumb.resize((100, 100), Image.ANTIALIAS)
		im.paste(thumb, box=(350, 150*i+50, 450, 150*i+150))

		thumb = Image.open(thumbs_y[results[i][2]])
		thumb = thumb.resize((100, 100), Image.ANTIALIAS)
		im.paste(thumb, box=(500, 150*i+50, 600, 150*i+150))		

	del draw
	im.save('groupings.png')
@profile
def opto3(conmat, thresh, tot):
	nat = np.arange(100)
	art = np.arange(100,200)
	ind = np.arange(64)
	ind2 = arange(32)

	natmat = conmat[:,:100]
	artmat = conmat[:,100:]
	artind = argsort(artmat, axis=1)
	natind = argsort(natmat, axis=1)
	uniq = zeros((64,200))
	artind = artind+100
	uniq[:,1::2]=natind
	uniq[:,::2]=artind
	count = 1
	contin = True
	f = lambda x, y : conmat[x][y]

	while contin:
		ranind = np.random.permutation(ind)
		ranind2 = np.random.permutation(ind)
		ranind3 = np.random.permutation(ind)
		ranind4 = ranind3[:32]
		ranind5 = ranind3[32:]
		rans4 =list(ranind4)
		rans5 =list(ranind5)
		rans2 =list(ranind2)
		rans = list(ranind)
		arr = zeros((64,3))
		arr2 = zeros((64,3))
		usedinds = set()
		for i in rans:
			k = 0
			j = 0
			while j <1:
				if str(artind[i][k]) not in usedinds:
					arr[i][j] = conmat[i][artind[i][k]]
					arr2[i][j] = artind[i][k]
					usedinds.add(str(artind[i][k]))
					j += 1
					k += 1
				else:
					k += 1
		for i in rans2:
			k = 0
			j = 1
			while j <2:
				if str(natind[i][k]) not in usedinds:
					arr[i][j] = conmat[i][natind[i][k]]
					arr2[i][j] = natind[i][k]
					usedinds.add(str(natind[i][k]))
					j += 1
					k += 1
				else:
					k += 1
		for i in rans4:
			k = 0
			j = 2
			while j <3:
				if str(natind[i][k]) not in usedinds:
					arr[i][j] = conmat[i][natind[i][k]]
					arr2[i][j] = natind[i][k]
					usedinds.add(str(natind[i][k]))
					j += 1
					k += 1
				else:
					k += 1
		for i in rans5:
			k = 0
			j = 2
			while j <3:
				if str(artind[i][k]) not in usedinds:
					arr[i][j] = conmat[i][artind[i][k]]
					arr2[i][j] = artind[i][k]
					usedinds.add(str(artind[i][k]))
					j += 1
					k += 1
				else:
					k += 1
		check2 = True
		withs = zeros((64))
		j=0
		for i in (arr2):
			within = (sim_mat[i[0]][i[1]]+sim_mat[i[0]][i[2]]+sim_mat[i[1]][i[2]])/3
			withs[j]=within
			j+=1
			if within > tot:
				check2 = False
				break
		sums = np.sum(arr, axis=1)
		checks = (sums/3.0) < thresh
		check = np.all(checks)
		if check and check2:
			print arr
			print withs
			return arr2
			contin = False
		count += 1
		if (count%100 == 0):
			print count

res = opto3(conmat, 0.33, 0.1)

#res = opto(conmat, 0.4, 0.47)
res = res.astype(int)
natcnt = 0
artcnt = 0
for row in res:
	a = row[0]>99
	b = row[1]>99
	c = row[2]>99
	if a and b and c:
		artcnt += 1
	if not a and not b and not c:
		natcnt += 1
print natcnt
print artcnt	

out = read_csv('csv/study.csv')

images = out[0]

obs = images[:200]
sce = images[200:]


draw_group(res, sce, obs)


#@m = Image.new('RGB', ((x+2)*50, (y+2)*50), (255, 255, 255))



# columns = [1, 2, 3]
# index = ind
 

# def chunks(l, n):
# 	chu = []
# 	for i in xrange(0, len(l), n):
# 		chu.append(l[i:i+n])
# 	return chu


# art = chunks(art, 12)
# nat = chunks(nat, 12)
# indchu = chunks(ind, 8)
# elems = []
# for i in range(0,len(indchu)):
# 	elems.append([art[i],nat[i]])

# for i in range(0,len(arr),8):
# 	j = i/8
# 	arr[i][0] = elems[j][0][0]
# 	arr[i][1] = elems[j][0][1]
# 	arr[i][2] = elems[j][0][2]
# 	arr[i+1][0] = elems[j][0][3]
# 	arr[i+1][1] = elems[j][0][4]
# 	arr[i+1][2] = elems[j][1][0]
# 	arr[i+2][0] = elems[j][0][5]
# 	arr[i+2][1] = elems[j][1][1]
# 	arr[i+2][2] = elems[j][0][6]
# 	arr[i+3][0] = elems[j][0][7]
# 	arr[i+3][1] = elems[j][1][2]
# 	arr[i+3][2] = elems[j][1][3]
# 	arr[i+4][0] = elems[j][1][4]
# 	arr[i+4][1] = elems[j][0][8]
# 	arr[i+4][2] = elems[j][0][9]
# 	arr[i+5][0] = elems[j][1][5]
# 	arr[i+5][1] = elems[j][0][10]
# 	arr[i+5][2] = elems[j][1][6]
# 	arr[i+6][0] = elems[j][1][7]
# 	arr[i+6][1] = elems[j][1][8]
# 	arr[i+6][2] = elems[j][0][11]
# 	arr[i+7][0] = elems[j][1][9]
# 	arr[i+7][1] = elems[j][1][10]
# 	arr[i+7][2] = elems[j][1][11]

for i in range(len(uniq)):
	firs = uniq[i][0:6]
	ranfirs = random.permutation(firs)
	uniq[i][0] = ranfirs[0]
	uniq[i][1] = ranfirs[1]
	uniq[i][2] = ranfirs[2]
	uniq[i][3] = ranfirs[3]
	uniq[i][4] = ranfirs[4]
	uniq[i][5] = ranfirs[5]



