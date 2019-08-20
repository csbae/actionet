
# coding: utf-8

# # Demo for Semantic Trajectory Analysis Algorithm

# #### List the files in the "data"directory to make sure that all the required files exist

# In[ ]:

ls -al ../data


# #### Import "pandas" library, which is for I/O

# In[ ]:

import pandas as pd


# #### Using pandas to read structured raw data with hilbert indexing

# In[ ]:

df = pd.read_csv('../data/mit_trj_parkinglot_all_hilbert.csv')


# #### Group data by doc_id (i.e. trajectory ID)

# In[ ]:

dfgs = df.groupby('doc_id')


# #### First 5 rows will be like: 

# In[ ]:

df.head()


# #### All column names:  

# In[ ]:

df.columns


# #### Trip point index as a document:

# In[ ]:

list(dfgs.groups.keys())[:2]


# In[ ]:

tmp = df.ix[dfgs.groups['md_27345']]['hilbert_idx']


# #### Index of a trip: 

# In[ ]:

tmp.values


# In[ ]:




# #### Import "gensim" library for document vector

# In[ ]:

from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence


# #### Generate labelled sentences (1. labelled 2. only contain clean words)

# In[ ]:

sentences = []
for k in dfgs.groups.keys():
    tmp = df.ix[dfgs.groups[k]]['hilbert_idx']
    sentences.append(LabeledSentence(words=[str(i) for i in tmp], tags=[k]))


# #### First sentence: 

# In[ ]:

sentences[1]


# #### Initialization of the model where alpha is set to a value and sentences are input

# In[ ]:

model = Doc2Vec(alpha=0.01)


# In[ ]:

model.build_vocab(sentences)


# #### Train  model: 

# In[ ]:

for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha


# #### After training, you can choose to save the model for future use

# In[ ]:

model.save('../data/trip_as_vec.model')


# #### Alternatively, you can skip the processes of initilization/training/saving model, but load a existing model instead

# In[ ]:

model = Doc2Vec.load('../data/trip_as_vec.model')


# #### Size of a document vector: 

# In[ ]:

model.docvecs[0].shape


# #### Import "matplotlib" for plotting diagram: 

# In[ ]:

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt


# #### Tag for second sentence: 

# In[ ]:

t = model.docvecs.index2doctag[1]
print(t)


# #### Document vector with tag above:

# In[ ]:

model.docvecs.indexed_doctags(t)


# #### Print all the information within one trajectory

# In[ ]:

tag = 'md_27345'
print('Original description {:s}'.format(tag))
idx = int(tag.split('_')[-1])
tmp = df.ix[dfgs.groups['md_27345']]
print(tmp)


# #### Define a method to plot a single trip

# In[ ]:

plt.figure(figsize=(10, 8))
def plot_trip(doc_id, lw=None, c=None, s=1):
    s = str("{0:.2f}".format(s * 100)) + "%"
    idx = dfgs.groups[doc_id]
    tmp = df.ix[idx]
    plt.plot(tmp['x'], 360 - tmp['y'], color=c, linewidth=lw, label=s)
    


# #### Read in the background image

# In[ ]:

img = plt.imread('../data/parkinglot.png')
img_gry = img[:, :, 0]


# #### Check the similarity of three trajectories base on one query trajectory

# In[ ]:

tag1 =  model.docvecs.index2doctag[100]
tag2 =  model.docvecs.index2doctag[35736]
tag3 =  model.docvecs.index2doctag[10000]
tag4 =  model.docvecs.index2doctag[1000]
plt.figure(figsize=(10, 8))
plt.imshow(img_gry[::-1, :], extent=(0, 480, 0, 360), cmap='gray', origin="lower")
plot_trip(tag1, lw=4, c='c')
plot_trip(tag2, lw=1, s=model.docvecs.similarity(tag1, tag2))
plot_trip(tag3, lw=1, s=model.docvecs.similarity(tag1, tag3))
plot_trip(tag4, lw=1, s=model.docvecs.similarity(tag1, tag4))
plt.legend()


# #### Find the most similar trips based on a query trip

# In[ ]:

tag =  model.docvecs.index2doctag[100]
tag = "md_33358"
plt.figure(figsize=(10, 8))
plt.imshow(img_gry[::-1, :], extent=(0, 480, 0, 360), cmap='gray', origin="lower")
plot_trip(tag, lw=4, c='c')
print('='*100)
count = 0
for t, s in model.docvecs.most_similar(tag):
    if count < 8:
        plot_trip(t, lw=1, s=s)
        plt.legend()
    count = count + 1


# In[ ]:




# ## THE DEMO FINISHES HERE!
