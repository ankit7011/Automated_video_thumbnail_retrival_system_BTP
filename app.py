from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from clustimage import Clustimage
from pyvis.network import Network
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import math
from keras import Model
from sklearn.decomposition import PCA
from mtcnn.mtcnn import MTCNN
import pickle as pk
from clustimage import Clustimage
import networkx as nx
from brisque import BRISQUE
from keras.applications.vgg16 import VGG16 
from keras.models import Model
import keras.utils as image
# from keras.preprocessing.image import load_img 
# from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def get_frames(path, num):
    
    if 'frames' not in os.listdir():
        os.mkdir('frames')

    vidObj = cv2.VideoCapture(path)
    count = 0  
    success = 1
    frames = []

    while success:
        success, image = vidObj.read()
        if not success:
            break

        if(count % num == 0):
            c = count / num
            cv2.imwrite('./frames/%d.jpg' % c, image)
            frames.append(image)
        count += 1

    vidObj.release()
    cv2.destroyAllWindows()
    
    return frames

def get_face_data(frames):

    faces = {}
    
    for i in range(len(frames)):
        faces[i] = []
        face_locations = detector.detect_faces(frames[i])
        
        if len(face_locations) == 0:
            continue

        for f in face_locations:
            x = f['box'][0]  
            y = f['box'][1]
            w = f['box'][2]
            h = f['box'][3]
            x1 = max(x-30, 0)
            x2 = min(x + w + 30, frames[i].shape[1])
            y1 = max(y - 30, 0)
            y2 = min(y + h + 30, frames[i].shape[0])
            t = frames[i][y1: y2, x1 : x2]
            faces[i].append(t[...,::-1])   
            
    return faces

def save_face(faces):
    
    if 'faces' not in os.listdir():
        os.mkdir('faces')

    os.chdir('./faces')

    for i in faces.keys():
        if faces[i] == []:
            continue

        c = 1
        for j in faces[i]:
            img_rgb = j[:,:,::-1]
            name = 'fr' + str(i) + '_f' + str(c) + '.png'
            c += 1
            cv2.imwrite(name, img_rgb)
        
    os.chdir('..')


def get_cluster(cl, path):
    
    a = cl.find(path)
    filename = [*a.keys()][1]
    b = list(a[filename]['labels'])
    return max(set(b), key = b.count)

def frame_data(cl, faces):

    frame = {}

    for i in faces:
        if faces[i] == []:
            continue

        frame[i] = {'no' : 0, 'faces' : []}
        frame[i]['no'] = len(faces[i])
        c = 1
        frame[i]['faces'] = {}

        for face in faces[i]:

            if c not in frame[i]['faces'].keys():
                frame[i]['faces'][c] = {'area' : 0, 'arr' : np.zeros(face.shape) , 'char' : -1 }

            frame[i]['faces'][c]['area'] = (face.shape[0] * face.shape[1]) / area_frame
            frame[i]['faces'][c]['arr'] = face
            c += 1

    for i in frame.keys():
        for j in range(1, frame[i]['no'] + 1):
            face_path = './faces/' + 'fr' + str(i) + '_f' + str(j) + '.png'
            c = get_cluster(cl, face_path)
            frame[i]['faces'][j]['cluster'] = c

    return frame


##PART 3 - CHARACTER INTERACTION GRAPH

def graph_weights(frame):
    graph_wts = {}
    for i in frame.keys():
        clusts = []
        for j in range(1, frame[i]['no'] + 1):
            clusts.append(frame[i]['faces'][j]['cluster'])

        res = [(a, b) for idx, a in enumerate(clusts) for b in clusts[idx + 1:]]
        for i in res:
            if i not in graph_wts.keys():
                graph_wts[i] = 0

            graph_wts[i] += 1
            
    return graph_wts

def create_graph(cl, char_prom, graph_wts):
    
    G = Network(notebook = True)

    for i in char_prom.keys():
        path = cl.results_unique['pathnames'][i]
        G.add_node(i, label = str(i), shape = 'image', image = path)

    for i in graph_wts.keys():
        s = i[0]
        d = i[1]
        if s == d:
            continue
        w = graph_wts[i]
        G.add_edge(s, d, value = w, title = str(w))
        

        G.show('templates\character_graph.html')
        

def get_pi(frame, graph_wts, char_prom):

    interactions = {}
    for i in graph_wts.keys():
        a = i[0]
        b = i[1]
        if a == b:
            continue

        if a not in interactions.keys():
            interactions[a] = 0
        if b not in interactions.keys():
            interactions[b] = 0

        interactions[a] += graph_wts[i]
        interactions[b] += graph_wts[i]
        
        
    scores_pi = []
    for i in frame.keys():
        num = 0
        sc = 0
        for j in range(1, frame[i]['no'] + 1):
            f = frame[i]['faces'][j]
            a = f['area']
            p = char_prom[f['cluster']]
            num += a * (p + interactions.get(f['cluster'], 0))

        sc = num / len(frame.keys())
        scores_pi.append((sc, i))

    return scores_pi

def get_ei(arr):
    
    i = Image.fromarray(arr)
    i = i.convert('L')
    i = i.resize((48, 48))
    n = np.asarray(i).reshape(1, 48, 48, 1)
    sc = ei_model.predict(n)
    return sc[0][0]

def ei_total(frame):

    scores_ei = []   
    for i in frame.keys():
        num = 0
        sc = 0
        for j in range(1, frame[i]['no'] + 1):
            f = frame[i]['faces'][j]
            a = f['area']
            arr = f['arr']
            ei = get_ei(arr)
            num += a * ei

        sc = num / len(frame.keys())
        scores_ei.append((sc, i))
    
    return scores_ei

# def predict_old(video_path):
    
#     print('-------Extracting frames--------')
#     frames = get_frames(video_path, num_frames)
#     print('-------Extracting face data------')
#     faces = get_face_data(frames)

#     with open('./faces.pkl', 'wb') as f:
#         pk.dump(faces, f)

#     print('--------Storing extracted faces-------')
#     save_face(faces)

#     ##PART 2 - Clustering
#     print('--------Clustering data-----------')
#     cl = Clustimage()
#     results = cl.fit_transform('./faces',cluster='agglomerative', evaluate='dbindex', linkage = 'centroid', max_clust = 20)

#     print('--------Extracting frame data-----------')
#     frame = frame_data(cl, faces)
#     with open('frame_data.pkl', 'wb') as f:
#         pk.dump(frame, f)
#     char_prom = dict(pd.value_counts(cl.results['labels']))

#     ##PART 3 - GRAPH
#     print('-------Creating Graph---------')
#     graph_wts = graph_weights(frame)
#     create_graph(cl, char_prom, graph_wts)

#     ##PART 4 - SCORE
#     print('-------Scoring Frames--------')
#     scores_pi = get_pi(frame, graph_wts, char_prom)
#     scores_ei = ei_total(frame)

#     scores = []
#     for i in range(len(scores_pi)):
#         s = (scores_pi[i][0] + 100 * scores_ei[i][0]) / 101
#         scores.append((s, scores_pi[i][1]))
    
#     a = sorted(scores, reverse = True)
#     t10 = []
#     m = min(16, len(a))
#     for i in a[:m]:
#         t10.append(i[1])

#     paths = []
#     for i in range(m):
#         # tpath = './frame/' + str(t10[i]) + '.jpg' # original by aniket
#         tpath = './images/' + str(t10[i]) + '.jpg'  # by ankit
#         paths.append(tpath)

#     print(paths)

#     print('-------Displaying top frames---------')
#     space = plt.figure(figsize = (20,20))     
#     for i in range(m):
#         ax = space.add_subplot(4,4,i+1)    
#         ax.imshow(frames[t10[i]][...,::-1])
#     trimmed_name = video_path[:-4]
#     plt.savefig('static/images/'+trimmed_name + ".png")
#     # plt.show()

#     return paths

def predict(frames,faces,video_path):
    
    print("i am in predict")

    ##PART 2 - Clustering
    print('--------Clustering data-----------')
    cl = Clustimage()
    results = cl.fit_transform('./faces',cluster='agglomerative', evaluate='dbindex', linkage = 'centroid', max_clust = 20)

    print('--------Extracting frame data-----------')
    frame = frame_data(cl, faces)
    with open('frame_data.pkl', 'wb') as f:
        pk.dump(frame, f)
    char_prom = dict(pd.value_counts(cl.results['labels']))

    ##PART 3 - GRAPH
    print('-------Creating Graph---------')
    graph_wts = graph_weights(frame)
    create_graph(cl, char_prom, graph_wts)

    ##PART 4 - SCORE
    print('-------Scoring Frames--------')
    scores_pi = get_pi(frame, graph_wts, char_prom)
    scores_ei = ei_total(frame)

    scores = []
    for i in range(len(scores_pi)):
        s = (scores_pi[i][0] + 100 * scores_ei[i][0]) / 101
        scores.append((s, scores_pi[i][1]))
    
    a = sorted(scores, reverse = True)
    t10 = []
    m = min(16, len(a))
    for i in a[:m]:
        t10.append(i[1])

    paths = []
    for i in range(m):
        # tpath = './frame/' + str(t10[i]) + '.jpg' # original by aniket
        tpath = './images/' + str(t10[i]) + '.jpg'  # by ankit
        paths.append(tpath)

    print(paths)

    print('-------Displaying top frames---------')
    space = plt.figure(figsize = (20,20))     
    for i in range(m):
        ax = space.add_subplot(4,4,i+1)    
        ax.imshow(frames[t10[i]][...,::-1])
    plt.savefig('output.jpg')
    trimmed_name = video_path[:-4]
    plt.savefig('static/images/'+trimmed_name + "char.png")
    return paths


#-----------------------Added Non - char -------------------


def extract_features(file, model):
    img = image.load_img(file, target_size=(224,224))
    img = np.array(img) 
    reshaped_img = img.reshape(1,224,224,3) 
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx, use_multiprocessing=True)
    return features


def no_char_data():
    
    data = {}
    for f in os.listdir('./no_char'):
        path = './no_char/' + f 
        feat = extract_features(path,model)
        data[f] = feat
        
    return data

def pca_data(data):
    
    filenames = np.array(list(data.keys()))
    feat = np.array(list(data.values()))
    feat = feat.reshape(-1,4096)
    # reduce the amount of dimensions in the feature vector
    pca = PCA(n_components=min(feat.shape[0], feat.shape[1]))
    x = pca.fit_transform(feat)
    
    return x

def cluster_data(x):
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(x)
    return kmeans

def create_dicts(data, kmeans):
    groups = {}
    filenames = np.array(list(data.keys()))
    for file, cluster in zip(filenames,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    img_to_clust = {}
    for g in groups.keys():
        for i in groups[g]:
            a = int(i.split()[0])
            img_to_clust[a] = g
        
    return groups, img_to_clust

def pi_score_nonchar(groups, img_to_clust,no_char):
    
    pi_scores = {}
    for i in no_char.keys():
        clust = img_to_clust[i]
        num = len(groups[clust])
        pi_scores[i] = num / len(no_char.keys())
        
    return pi_scores

def li_score(no_char):
    
    li_scores = {}
    for i in no_char.keys():
        s = obj.score(no_char[i])
        if math.isnan(s):
            s = 150
        li = 1 - s/150
        li_scores[i] = li
    
    return li_scores

def predict_nochar(no_char,frames,video_path):
    print("i am in non-char-predict")
    data = no_char_data()
    x = pca_data(data)
    kmeans = cluster_data(x)
    groups, img_to_clust = create_dicts(data, kmeans)
    pi_scores = pi_score_nonchar(groups, img_to_clust,no_char)
    li_scores = li_score(no_char)
    
    scores = []

    for i in li_scores.keys():
        l = li_scores[i]
        p = pi_scores[i]
        s = (l + 10*p) / 11
        scores.append((s, i))
        

    a = sorted(scores, reverse = True)
    t10 = []
    m = min(16, len(a))
    for i in a[:m]:
        t10.append(i[1])

    paths = []
    for i in range(m):
        tpath = './frame/' + str(t10[i]) + '.jpg'
        paths.append(tpath)

    print(paths)

    print('-------Displaying top frames---------')
    space = plt.figure(figsize = (20,20))     
    for i in range(m):
        ax = space.add_subplot(4,4,i+1)    
        ax.imshow(frames[t10[i]][...,::-1])
    plt.savefig('output_nochar.jpg')
    trimmed_name = video_path[:-4]
    plt.savefig('static/images/'+trimmed_name + "non_char.png")

    return paths

    

def basic(video_path, num_frames):
    print("i am in basic")
    print('-------Extracting frames--------')
    frames = get_frames(video_path, num_frames)
    print('-------Extracting face data------')
    faces = get_face_data(frames)

    with open('./faces.pkl', 'wb') as f:
        pk.dump(faces, f)


    # with open('./faces.pkl', 'rb') as f:
    #     faces = pk.load(f)

    print('--------Storing extracted faces-------')
    save_face(faces)

    no_char = {}
    for i in faces.keys():
        if faces[i] == []:
            no_char[i] = frames[i]

    if 'no_char' not in os.listdir():
        os.mkdir('no_char')
    os.chdir('no_char')

    for i in no_char.keys():
        name = str(i) + ' .png'
        cv2.imwrite(name, no_char[i])

    os.chdir('..')

    return frames, faces, no_char



# #INIT VARIABLES
basepath = os.getcwd()
ei_model = tf.keras.models.load_model(basepath + '/ei_score.h5')
num_frames = 50


face_locations = []
detector = MTCNN()
area_frame = 720 * 1080

#new 

model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
obj = BRISQUE(url=False)



#--------------------------------------------------------------------Flask--------------------------------------------------------

from flask import Flask, flash, request, redirect, url_for, render_template
import os 

app = Flask(__name__)

UPLOAD_FOLDER = 'static/videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
   return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    frames, faces, no_char = basic(filename, num_frames)
    paths = predict(frames,faces,filename)
    paths_nochar = predict_nochar(no_char,frames,filename)

    print("My New Path is ",filename) 
    trimmed_name_char = filename[:-4] + 'char.png'
    trimmed_name_nonchar = filename[:-4] + 'non_char.png'
    filenames=[trimmed_name_char,trimmed_name_nonchar]
    return render_template('play.html', filenames=filenames)

@app.route('/character_graph')
def example():
    return render_template('character_graph.html')    

if __name__ == '__main__':
   app.run(debug=True)