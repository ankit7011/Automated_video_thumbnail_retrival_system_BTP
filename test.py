from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import cv2
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pk
from clustimage import Clustimage
from pyvis.network import Network
import tensorflow as tf
from mtcnn.mtcnn import MTCNN


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
        
        G.show('example.html')
        

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

def predict(video_path):
    
    print('-------Extracting frames--------')
    frames = get_frames(video_path, num_frames)
    print('-------Extracting face data------')
    faces = get_face_data(frames)

    with open('./faces.pkl', 'wb') as f:
        pk.dump(faces, f)

    print('--------Storing extracted faces-------')
    save_face(faces)

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
        tpath = './frame/' + str(t10[i]) + '.jpg'
        paths.append(tpath)

    print(paths)

    print('-------Displaying top frames---------')
    space = plt.figure(figsize = (20,20))     
    for i in range(m):
        ax = space.add_subplot(4,4,i+1)    
        ax.imshow(frames[t10[i]][...,::-1])
    plt.show()

    return paths





#INIT VARIABLES
basepath = os.getcwd()
ei_model = tf.keras.models.load_model(basepath + '/ei_score.h5')
num_frames = 100
video_path = basepath + '/test.mp4'
if 'Thumbnail Data' not in os.listdir():
    os.mkdir('Thumbnail Data')
os.chdir('Thumbnail Data')
face_locations = []
detector = MTCNN()
area_frame = 720 * 1080

paths = predict(video_path)
print('Finished')
