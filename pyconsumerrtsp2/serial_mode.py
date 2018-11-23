# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:39:10 2018
multisubject running on serial mode
@author: wtc10
"""


import cv2, pickle, glob, time  
from sklearn.externals import joblib   
import numpy as np 
import scipy.signal as sig
#from data_collect import visualize
import multiprocessing
import subprocess

import datetime
from PIL import ImageDraw, ImageFont, Image, ImageColor

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') 
xs=cv2.getTextSize('20-30', cv2.FONT_HERSHEY_SIMPLEX, 1, 0)

class face_tmp:
    def __init__(self, pts, idx):
        self.id=idx
        self.pts=pts
        self.mode=0
        self.time=[]
        self.green=[]
        self.age=np.zeros((1, 4))
        self.gender=np.zeros((1, 2))
        self.race=np.zeros((1, 4))
        self.last=0.0
        self.genderS="Unknown"
        self.raceS="Unknown"
        self.emotionS="Unknown"
        self.ageS="Unknown"
        self.emo_label="Unknown"
        self.avi_label=[0,0]
        self.attn=0
        self.color=(0,0,0)
       #self.avi=-1
    def clear(self):
        self.pts*=0.
        self.mode=0
    def on_mode(self, age, gender, race, emo, avi):
        self.mode=1
        self.age+=age
        self.gender+=gender
        self.race+=race
        self.emotion=emo
        self.avi=avi[0:1, :]/max(1, sum(avi[0, :]**2)**0.5)

        
class collect_tmp:
    def __init__(self):
        self.attn=[]
        self.emotion=np.zeros((1,7))
        self.time=[]
        self.hr=[]
        self.avi=np.zeros((1,2))
#        self.age_list=['20-30', '30-40', '40-60', '60+']
        self.gender_list=['F', 'M']
        self.race_list=['East Asian', 'European', 'South Asian', 'S-East Asian']
        self.emotion_list=['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.log_time=[]
        self.anchor=np.array([25.,35.,55.,75.])
    def on_mode(self, face_tp):
#        self.age=self.age_list[np.argmax(face_tp.age)]
        self.age='%d'%int(np.sum(self.anchor*face_tp.age[0,:])/np.sum(face_tp.age[0,:]))
        self.gender=self.gender_list[np.argmax(face_tp.gender)]
        self.race=self.race_list[np.argmax(face_tp.race)]
        self.emotion=np.concatenate((self.emotion, face_tp.emotion), axis=0)
        self.emo_label=self.emotion_list[np.argmax(np.mean(self.emotion[-10:,:], axis=0))]
        if self.race=='EU': self.race='EA'
        self.avi=np.concatenate((self.avi, face_tp.avi), axis=0)
        self.avi_label=np.mean(self.avi[-10:,:], axis=0)
        self.log_time.append(face_tp.time[-1])
    def update_attn(self, attn):
        self.attn.append(attn)

def faceCrop(img, existing=[],cropscale=(1,1)): # cut the face with viola-jones haar cascade 
    #img=cv2.imread(path, 0)#Image.open(img)
    new_faces = faceCascade.detectMultiScale(img, scaleFactor = 1.2, minNeighbors = 2,
                                          minSize = (30,30), flags = 0) # Minsize -(30,30) Min size for the face is here ARYEL
    
    if isinstance(new_faces, tuple):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2=cv2.equalizeHist(img2)
        new_faces = faceCascade.detectMultiScale(img2, scaleFactor = 1.2, minNeighbors = 2,
                                          minSize = (int(img.shape[1]*0.15),int(img.shape[0]*0.15)), flags = 0)
       
    if not isinstance(faces, tuple):
        clear_faces=clear_boxes(existing, new_faces)
        return [imCrop(ii, cropscale) for ii in clear_faces]
        
#        return list(faces[0]), img[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2], :]
    else:
        return []
    
def no_intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: 
      return True # or (0,0,0,0) ?
  return False

def clear_boxes(faces, new_faces):
    existing=[]
    coll=[]
    for ii in faces:
        existing.append(np.concatenate([np.min(ii.pts, axis=0), np.max(ii.pts, axis=0)-np.min(ii.pts, axis=0)]))
    for jj in new_faces:
        try:
            for ii in existing:
               assert no_intersection(jj, ii)
            for ii in coll:
               assert  no_intersection(jj, ii)
        except:
            continue
        coll.append(jj.reshape((1, -1)))
    if coll:
        return np.vstack(coll)
    else:
        return []
    
def imCrop(cropBox, boxScale=(1, 1)):# cut out the face image if it was there
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]
    # Calculate scale factors
    yDelta=cropBox[3]*(boxScale[1]-1)
    box_down=int(cropBox[1]+cropBox[3]+yDelta)
    return [cropBox[0], cropBox[1],cropBox[0]+cropBox[2], box_down]

def reverse_point(pts, start_point, scale):
    pts/=scale
    pts[:, 0]+=start_point[0]
    pts[:, 1]+=start_point[1]
    return pts

def point_calc(config, bb):
    img_shape=config['img_shape'] #variable to change, size of the image to work on  
    portion=config['portion'] #variable to change, vertical protion to cut 
    stretch=config['stretch']
    start_point=bb[0:2]
    start_point[1]+=np.round(bb[3]*portion[0]/stretch).astype('int64')
    scale=img_shape/float(bb[2])
    return start_point, scale

        
def create_box(tracker_config, mmx, mmy, dis):
    stretch=tracker_config['stretch']
    left=mmx-dis*stretch[0]
    right=mmx+dis*stretch[1]
    upp=mmy-dis*stretch[2]
    down=mmy+dis*stretch[3]
    return np.array([left, upp, right, down])

def norm_dis(tracker_config, target):
#        eye_point=np.array(eye_point)
    eye_point=np.array(tracker_config['eye_point'])
#        no_point=target.shape[1]//2
    mmX=np.mean(target[eye_point,0], axis=0)
    mmY=np.mean(target[eye_point,1], axis=0)
    dx=target[eye_point, 0]-mmX
    dy=target[eye_point, 1]-mmY
    return mmX, mmY, np.mean((dx**2+dy**2)**0.5)

def tracker_calc_hog(tracker_config, tracker_pad, pts_c, idx, img, hogger): 
    img_size=tracker_config['face_size'][idx]
    Delta=img_size[0]//2
    pts_c=pts_c+tracker_pad
    pts_c[:, 0]=np.clip(pts_c[:,0], tracker_pad, img.shape[1]-tracker_pad)-Delta
    pts_c[:, 1]=np.clip(pts_c[:,1], tracker_pad, img.shape[0]-tracker_pad)-Delta
    pts_c=tuple(map(tuple, np.round(pts_c).astype('int')))
    return np.array(hogger.compute(img, locations=pts_c)).astype('float').reshape((1, -1))

def procrustes(X, Y, scaling=True, reflection='best'):
    n,m = X.shape
    ny,my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    if reflection is not 'best':
        have_reflection = np.linalg.det(T) < 0
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T) 
    tform = {'rotation':T, 'scale':b, 'translation':c}
    return d, Z, tform

def inv_procrustes(X, trs):
    return np.dot((1/trs['scale'])*trs['rotation'], X.T-np.tile(trs['translation'], (X.shape[0],1)).T).T
 
def get_activations(X):
        clf=tracker_models[0]
        hidden_layer_sizes = clf.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)
        layer_units = [X.shape[1]] + hidden_layer_sizes + \
            [clf.n_outputs_]
        activations = [X]
        for i in range(clf.n_layers_ - 1):
            activations.append(np.empty((X.shape[0],
                                         layer_units[i + 1])))
        clf._forward_pass(activations)
        return activations[1]       

def HR_feature(timer, green):
    fs=30
    dur=20
    start=timer[-1]-20
    start_idx=np.nonzero(timer<=start)[0][-1]
#    assert(len(timer)-start_idx>dur*fs/2)
    even_times=np.linspace(start,timer[-1], dur*fs+1)
    interpolated = np.interp(even_times, timer[start_idx:], green[start_idx:]-np.mean(green[start_idx:]))
    freq, res=sig.welch(interpolated, fs=fs, nfft=601)
    upper,lower=50, 120
    bands=[ii.argmin() for ii in map(lambda x: abs(freq-x), [ii/60. for ii in [upper, lower]])]
    periodogram=res[ :bands[-1]]
    freq=freq[:bands[-1]]*60
    feature=np.log10(periodogram).reshape((1, -1))
    return freq,feature

def estimate_projection(img_points, model_points, height):
    model_points=np.hstack([model_points, np.ones((model_points.shape[0], 1))])
    img_points[:, 1]=height-img_points[:, 1]
    npt=img_points.shape[0]
    A=np.zeros((2*npt, 8))
    for ii in range(npt): 
        A[2*ii, 0:4]=model_points[ii, :]
        A[2*ii+1, 4:8]=model_points[ii, :]
#    A[:, [3, 7]]=1
    b=img_points.reshape((-1, 1))
#    x=np.dot(np.linalg.pinv(A), b)
    x=np.linalg.lstsq(A, b)[0]
    X=x.reshape((2, -1))
    R=np.zeros((3,3))
    R[0:2, :]=X[:, 0:3]
    R[0, :]/=np.linalg.norm(R[0, :])
    R[1, :]/=np.linalg.norm(R[1, :])
    R[2, :]=np.cross(R[0, :], R[1, :])
    U, ss, V = np.linalg.svd(R, full_matrices=True)
    V = V.T
    T = np.dot(V, U.T)
    assert abs(1-np.linalg.det(T))<1e-6
    return (T[-1, -1]-0.5)*2

def save_worker(input_queue): 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('collect.avi',fourcc, 10.0, (1920,1080))
    while True:
        try:
            img=input_queue.get()
            input_queue.task_done()
            if img.any():
                out.write(img)
            else:    
                break
        except:
            break
    out.release()

def distanceXY(aPoint1,aPoint2):
    distance=(aPoint1[0]-aPoint2[0])*(aPoint1[0]-aPoint2[0])
    distance+=(aPoint1[1]-aPoint2[1])*(aPoint1[1]-aPoint2[1])
    distance=np.sqrt(distance)
    return distance

def GroupSensing(aFaceList, translate, faceImgList, or_img, aLanguage=False):
    colours=[(0,255,255),(255,0,255),(255,0,0), (0,255,0), (0,0,255),(255,255,0),(255,255,255)]
    index=0
    col=0
    img = np.ones((len(or_img),int(len(or_img[1])),3), np.uint8)*0
    overall_valence=0
    overall_arousal=0
    #Sanity check
    if len(aFaceList) != len(faceImgList):
        return img, or_img
    if len(aFaceList)<1:
        return img, or_img

    #Order the faces from left to right
    tupple = sorted(zip(aFaceList,faceImgList),key=lambda pair:np.min(pair[0].pts[:,0]))
    aFaceList, faceImgList = zip(*tupple)
    #Determine the language
    if aLanguage==False:
        lang = 0 #English
    else:
        lang = 1 #Japanese
    #Load font type for Japanese/English text    
    font=ImageFont.truetype("fonts\Arial-Unicode-Regular.ttf",25)
    #Height setting of text
    yPos1 = 115 #For individual analysis
    yPos2 = yPos1 + 230 #For group analysis
    xPos1 = 140 #For individual analysis
    xPos2 = 250 #For group analysis
    xPosLegend = 10 #For legend
    height = 30 #Height of text
    halfHeight = 20
    
    for face,faceImg in zip(aFaceList, faceImgList):
        overall_valence+=face.avi_label[0]
        overall_arousal+=face.avi_label[1]
        face.distance=distanceXY(face.pts[12, :],face.pts[21,:])

#        cv2.putText(img, face.genderS, (140+index,60), cv2.FONT_HERSHEY_SIMPLEX, fontSize, colours[col])
#        cv2.putText(img, face.raceS, (140+index,90), cv2.FONT_HERSHEY_SIMPLEX, fontSize, colours[col])
#        cv2.putText(img, face.ageS, (140+index,120), cv2.FONT_HERSHEY_SIMPLEX, fontSize, colours[col])
#        cv2.putText(img, face.emo_label, (140+index,150), cv2.FONT_HERSHEY_SIMPLEX, fontSize, colours[col])
#        cv2.putText(img, face.ageS, (xPos1+index,yPos1+height*3-5), cv2.FONT_HERSHEY_SIMPLEX, fontSize, colours[col])

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((xPos1+index,yPos1+height*0), translate(face.genderS,lang), font=font, fill=colours[col])
        draw.text((xPos1+index,yPos1+height*1), translate(face.raceS,lang), font=font, fill=colours[col])
        #draw.text((xPos1+index,yPos1+height*2), translate(face.ageS,lang), font=font, fill=colours[col])
        draw.text((xPos1+index,yPos1+height*2), face.ageS, font=font, fill=colours[col])
        draw.text((xPos1+index,yPos1+height*3), translate(face.emo_label,lang), font=font, fill=colours[col])
        img = np.array(img_pil)

        #Superimpose face image onto background image
        xCoord = xPosLegend
        yCoord = xPos1+index
        img[xCoord:xCoord+100,yCoord:yCoord+100,:] = faceImg[:, :, np.newaxis]

        x=int((face.avi_label[0]+1)*50+140+index)
        cv2.circle(img,(x,yPos1+height*4+halfHeight), 5, colours[col], -1)
        cv2.line(img,(xPos1+index,yPos1+height*4+halfHeight),(xPos1+100+index,yPos1+height*4+halfHeight),(255,255,255),1)

        x=int((face.avi_label[1]+1)*50+140+index)
        cv2.circle(img,(x,yPos1+height*5+halfHeight), 5, colours[col], -1)
        cv2.line(img,(xPos1+index,yPos1+height*5+halfHeight),(xPos1+100+index,yPos1+height*5+halfHeight),(255,255,255),1)
        
        x1=int(np.max(face.pts[:,0]))
        y1=int(np.max(face.pts[:,1]))
        x2=int(np.min(face.pts[:,0]))
        y2=int(np.min(face.pts[:,1]))
        
        cv2.rectangle(or_img,(x1,y1),(x2,y2),colours[col],1)
        face.color=colours[col]

        x=int(face.attn*100+(140+index))
        cv2.circle(img,(x,yPos1+height*6+halfHeight), 5, colours[col], -1)
        cv2.line(img,(xPos1+index,yPos1+height*6+halfHeight),(xPos1+100+index,yPos1+height*6+halfHeight),(255,255,255),1)
                
        index+=180
        col+=1
        if col>=len(colours):
            col=0
            
    mod_or_img = or_img.copy()
    colour=col
    groups=defineGroups(faces)
    for j in groups:
        mod_or_img = groupAnalysis(translate,lang,mod_or_img,j,colours[colour])
        mod_or_img = drawGroupRectangle(translate,lang,mod_or_img,j,colours[colour])
        colour+=1
        if colour>=len(colours):
            colour=0
    _,mod_or_img = findLeader(translate,lang,mod_or_img,faces,colours[colour],True)     

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((xPosLegend, yPos1+height*0), translate("Gender",lang), font=font)
    draw.text((xPosLegend, yPos1+height*1), translate("Race",lang), font=font)
    draw.text((xPosLegend, yPos1+height*2), translate("Age",lang), font=font)
    draw.text((xPosLegend, yPos1+height*3), translate("Emotion",lang), font=font)
    draw.text((xPosLegend, yPos1+height*4), translate("Arousal",lang), font=font)  
    draw.text((xPosLegend, yPos1+height*5), translate("Valence",lang), font=font)
    draw.text((xPosLegend, yPos1+height*6), translate("Attention",lang), font=font)

    draw.text((xPosLegend + 265, yPos2+height*0), translate("GroupAnalysis",lang), font=font)
    draw.text((xPosLegend, yPos2+height*1), translate("GroupHarmony",lang), font=font)
    draw.text((xPosLegend, yPos2+height*2), translate("GroupLeader",lang), font=font)
    draw.text((xPosLegend, yPos2+height*3), translate("GroupArousal",lang), font=font)
    draw.text((xPosLegend, yPos2+height*4), translate("GroupValence",lang), font=font)
    img = np.array(img_pil)


    cv2.rectangle(img,(xPosLegend-5,yPos2),(600,500),colours[5],1)
   
    leader,mod_or_img=findLeader(translate,lang,mod_or_img,aFaceList,(255,255,255),True)
    cv2.circle(img,(xPos2+20,yPos2+height*2+halfHeight),10,leader.color,5)
    
    overall_valence=(overall_valence/len(aFaceList))
    overall_arousal=(overall_arousal/len(aFaceList))
        
    x=int((overall_arousal+1)*100+250)
    cv2.circle(img,(x,yPos2+height*3+halfHeight), 5, leader.color, 1)
    cv2.line(img,(xPos2,yPos2+height*3+halfHeight),(xPos2+200,yPos2+height*3+halfHeight),(255,255,255),1)
 
    x=int((overall_valence+1)*100+250)
    cv2.circle(img,(x,yPos2+height*4+halfHeight), 5, leader.color, 1)
    cv2.line(img,(xPos2,yPos2+height*4+halfHeight),(xPos2+200,yPos2+height*4+halfHeight),(255,255,255),1)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((xPos2, yPos2+height*1), translate(groupSynchrony(aFaceList),lang), font=font)
    img = np.array(img_pil)
#    cv2.putText(img, groupSynchrony(aFaceList), (250,305), cv2.FONT_HERSHEY_SIMPLEX, 0.7, leader.color)
    
    return img, mod_or_img


class translator():
    def __init__(self):
        self.eng2jap = {}
        self.eng2jap['Gender'] = ["Gender:",u"性別:"]
        self.eng2jap['Race'] = ["Race:",u"人種:"]
        self.eng2jap['Age'] = ["Age:",u"年齢:"]
        self.eng2jap['Emotion'] = ["Emotion:",u"感情:"]
        self.eng2jap['Arousal'] = ["Arousal:",u"覚醒:"]
        self.eng2jap['Valence'] = ["Valence:",u"好感度:"]
        self.eng2jap['Attention'] = ["Attention:",u"関心度:"]
        
        self.eng2jap['GroupAnalysis'] = ["Group Analysis:",u"グループ分析"]
        self.eng2jap['GroupHarmony'] = ["Group Harmony:",u"調和:"]
        self.eng2jap['GroupLeader'] = ["Group Leader:",u"リーダー:"]
        self.eng2jap['GroupArousal'] = ["Group Arousal:",u"覚醒:"]
        self.eng2jap['GroupValence'] = ["Group Valence:",u"好感度:"]

        self.eng2jap['Female'] = ["Female",u"女性"]
        self.eng2jap['Male'] = ["Male",u"男性"]
        self.eng2jap['East Asian'] = ["East Asian",u"東アジア"]
        self.eng2jap['European'] = ["European",u"白人"]
        self.eng2jap['South Asian'] = ["South Asian",u"南アジア"]
        self.eng2jap['S-East Asian'] = ["S-East Asian",u"東南アジア"]

        self.eng2jap['Anger'] = ["Anger",u"怒り"]
        self.eng2jap['Disgust'] = ["Disgust",u"嫌悪"]
        self.eng2jap['Fear'] = ["Fear",u"恐れ"]
        self.eng2jap['Happy'] = ["Happy",u"喜び"]
        self.eng2jap['Neutral'] = ["Neutral",u"ニュートラル"]
        self.eng2jap['Sad'] = ["Sad",u"悲しみ"]
        self.eng2jap['Surprise'] = ["Surprise",u"驚き"]

        self.eng2jap['Influencer'] = ["Influencer",u"インフルエンサー"]
        self.eng2jap['Colleagues'] = ["Colleagues",u"同僚"]
        self.eng2jap['Harmonious'] = ["Harmonious",u"友好的な関係"]
        self.eng2jap['Conflictual'] = ["Conflictual",u"敵対的な関係"]
        
        self.eng2jap['Unknown'] = ['','']
        self.eng2jap['20-30'] = ['20-30','20-30']
        self.eng2jap['30-40'] = ['30-40','30-40']
        self.eng2jap['40-60'] = ['40-60','40-60']
        self.eng2jap['60+'] = ['60+','60+']
        
    def __call__(self,text,lang):
        return self.eng2jap[text][lang]
    

def drawGroupRectangle(translate,lang,mod_or_img,aFaceList, aColour):
    #find X_ MIN/MAX
    if len(aFaceList)>1:
        x1=int(np.max(aFaceList[0].pts[:,0]))
        y1=int(np.max(aFaceList[0].pts[:,1]))
        x2=int(np.min(aFaceList[0].pts[:,0]))
        y2=int(np.min(aFaceList[0].pts[:,1]))
        for i in aFaceList[1:]:
            x1=max(x1,int(np.max(i.pts[:,0])))+60
            y1=max(y1,int(np.max(i.pts[:,1])))+60
            x2=min(x2,int(np.min(i.pts[:,0])))-60
            y2=min(y2,int(np.min(i.pts[:,1])))-60
#        cv2.rectangle(or_img,(x1,y1),(x2,y2),aColour,len(aFaceList))
        img_pil = Image.fromarray(mod_or_img)
        draw = ImageDraw.Draw(img_pil,'RGB')
        draw.rectangle(((x1,y1),(x2,y2)), outline=aColour)
        mod_or_img = np.array(img_pil)  
    #cv2.putText(or_img, "colleagues", (x1,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))
    return mod_or_img

def groupType(translate,lang,mod_or_img,aFaceList, aColour,middleP):
   #type="Colleagues" 
   #TO DO
   position=(int(middleP[0]/len(aFaceList)-30),int(middleP[1]/len(aFaceList)-80))
   
   font=ImageFont.truetype("fonts\Arial-Unicode-Regular.ttf",25)
   img_pil = Image.fromarray(mod_or_img)
   draw = ImageDraw.Draw(img_pil)
   draw.text(position, translate("Colleagues",lang), font=font, fill=aColour)
   mod_or_img = np.array(img_pil)
#   cv2.putText(or_img, "Colleagues", position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, aColour)
   
   return mod_or_img

def defineGroups(aFaceList):
    groups=[]
    if len(aFaceList)>0:
        groups.append([aFaceList[0]])
        for i in range(1, len(aFaceList)):
            found=False
            for j in range(len(groups)):
                for k in range(len(groups[j])):
                    if faces[i]==groups[j][k]:
                        pass
                    else:
                        dist1= distanceXY(faces[i].pts[6, :],groups[j][k].pts[6, :])
                        dist2= abs(aFaceList[i].distance-aFaceList[j].distance)
                        if(dist1<800 and dist2<100):
                            groups[j].append(aFaceList[i])
                            found=True
                            break
                if not found:
                    groups.append([aFaceList[i]])
    return groups
                    
def groupAnalysis(translate,lang,mod_or_img,aFaceList,aColour):
    if len(aFaceList)>1:
        middleP=[0, 0]
        genders=0
        for k in aFaceList:
            middleP+=k.pts[6,:]
            genders+=np.argmax(k.gender)
        middleP1=(int(middleP[0]/len(aFaceList)-30),int(middleP[1]/len(aFaceList)-50))
        _,mod_or_img = findLeader(translate,lang,mod_or_img,aFaceList,aColour)
        mod_or_img = groupType(translate,lang,mod_or_img,aFaceList,aColour, middleP)
        #try:
        
        font=ImageFont.truetype("fonts\Arial-Unicode-Regular.ttf",25)
        img_pil = Image.fromarray(mod_or_img)
        draw = ImageDraw.Draw(img_pil)
        draw.text(middleP1, translate(groupSynchrony(aFaceList),lang), font=font, fill=aColour)
        mod_or_img = np.array(img_pil)
#        cv2.putText(or_img, groupSynchrony(aFaceList), middleP1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, aColour)
        
        #except:
            #print "Error in Group Synchrony"
    
    return mod_or_img    
        
def groupSynchrony(aFaceList):
    valenceGS=[]
    stdDev=0.5
    for i in aFaceList:
        valenceGS.append(i.avi_label[1])
    stdDev=np.std(valenceGS)
    if stdDev<0.3:
        return "Harmonious"
    if stdDev<0.6:
        return "Neutral"
    else:
        return "Conflictual"
   
def findLeader(translate,lang,mod_or_img,aFaceList,aColour,overal=False):
    genders=0
    leader=aFaceList[0]
    #oldestFace=aFaceList[0]
    oneGender=len(aFaceList)-genders
    for i in aFaceList:
        genders+=np.argmax(i.gender)
        if i.age[0][3]>leader.age[0][2]:
            leader=i
    #Method 1: One member is of different gender
        if oneGender==1 or oneGender==(len(aFaceList)-1):
            if np.argmax(i.gender)==0:
                leader=i
    if not overal:
        font=ImageFont.truetype("fonts\Arial-Unicode-Regular.ttf",25)
        img_pil = Image.fromarray(mod_or_img)
        draw = ImageDraw.Draw(img_pil)
        draw.text(tuple(leader.pts[6, :].astype('int')), translate("Influencer",lang), font=font, fill=aColour)
        mod_or_img = np.array(img_pil)
#        cv2.putText(or_img, "Influencer", tuple(leader.pts[6, :].astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 0.7, aColour)
    else:
        #print "Found"
        cv2.rectangle(img,(250,320),(260,330),leader.color,5)
    return leader, mod_or_img
        

# 0 = female # 1 =male # 2 =FemaleG #3=MaleG TO DO use dic
def displayCommercial(aGender):
    path="commercials/"
    if aGender==0:
        path=path+"female.png"
    elif aGender==1:
        path=path+"male.png"
    elif aGender==2:
        path=path+"femaleG.png"
    elif aGender==3:
        path=path+"maleG.png"
    else:
        path=path+"mixedG.png"
    
    com = cv2.imread(path,1)
    return com



if __name__=='__main__':
    folder='detector_allinone-trial-run-2018_02_21_16_42_59'
    tracker_folder='tracker_allinone-trial-run-2018_02_21_16_42_59'
    average=pickle.load(open('.\\detector_models_lr\\'+folder+'\\average_neutral.pkl', 'rb'))
    tracker_average=pickle.load(open('.\\tracker_models_lr\\'+tracker_folder+'\\average_neutral.pkl', 'rb'))
    classifier_model=joblib.load('.\\classifier_log\\%s\\classifier.pkl'%tracker_folder)
    models=[]
    for ii in glob.glob('.\\detector_models_lr\\'+folder+'\\model*.pkl'):
        models.append(joblib.load(ii))
    config=pickle.load(open('.\\detector_log\\'+folder+'\\config.pkl', 'rb'))
    tracker_models=[]
    for ii in glob.glob('.\\tracker_models_lr\\'+tracker_folder+'\\model*.pkl'):
        tracker_models.append(joblib.load(ii))
    
    tracker_config=pickle.load(open('.\\tracker_log\\'+tracker_folder+'\\config.pkl', 'rb'))    
    n_point=average.shape[0]
    pt_sel=[2,3,6]
    td_mean=pickle.load(open('.\\demo_log\\3d_mean.pkl', 'rb'))
    
    agemodel=joblib.load('.\\demo_log\\%s\\age.pkl'%tracker_folder)
    gendermodel=joblib.load('.\\demo_log\\%s\\gender.pkl'%tracker_folder)
    racemodel=joblib.load('.\\demo_log\\%s\\rrace.pkl'%tracker_folder)
    emotionmodel=joblib.load('.\\demo_log\\%s\\rexpression.pkl'%tracker_folder)
    avimodel=joblib.load('.\\demo_log\\%s\\avi.pkl'%tracker_folder)
    hrmodel=joblib.load('.\\demo_log\\%s\\tree.pkl'%tracker_folder)
    
    input_queue=multiprocessing.JoinableQueue()
    
    p = multiprocessing.Process(target=save_worker, args=(input_queue,))
    
    p.start()
    
    nbins=config['nbins']
    #    block_size=config.get('block_size', (2,2))
    de_hog=[]
    de_iter=len(config['face_size'])
    for idx in range(de_iter):
        block_size=config.get('block_size', ((2,2),(2,2),(2,2),(2,2)))[idx]
        img_size=config['face_size'][idx]
        cell_size=img_size//block_size[0] # unpack from parcel, for parallel computation 
        hog=cv2.HOGDescriptor(_winSize=(img_size[1] // cell_size[1] * cell_size[1],
                                      img_size[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
        de_hog.append(hog)
    
    outer=config['face_size'][0][0]
    tracker_pad=outer//2+outer%2
    stretch=tracker_config['stretch']
    ss=(stretch[2]+stretch[3])/(stretch[0]+stretch[1])
    tr_img=np.zeros((int(tracker_config['img_shape'])+2*tracker_pad, \
                               int(tracker_config['img_shape']*ss)+2*tracker_pad), dtype='uint8')
    #    tr_block_feature=tracker_config['nbins']*tracker_config['block_size'][0][0]**2
    #    tr_feature=np.zeros((1,n_point*tr_block_feature))
    
    nbins=tracker_config['nbins']
    #    block_size=tracker_config.get('block_size', (2,2))
    tr_hog=[]
    tr_iter=len(tracker_config['face_size'])
    for idx in range(tr_iter):
        block_size=tracker_config.get('block_size', ((2,2),(2,2),(2,2),(2,2)))[idx]
        img_size=tracker_config['face_size'][idx]
        cell_size=img_size//block_size[0] # unpack from parcel, for parallel computation 
        hog=cv2.HOGDescriptor(_winSize=(img_size[1] // cell_size[1] * cell_size[1],
                                      img_size[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
        tr_hog.append(hog)
    
    current=np.zeros((n_point, 2))
    id_dis=0
    pool_dict={}
#    cap=cv2.VideoCapture("couple.mp4")
    toggle_mode=0
    faces=[]
    now=time.time()
    thresh=0.
    del_list=list()
    cap=cv2.VideoCapture(0)
#    cap=cv2.VideoCapture('17.mp4')
    cap.set(3,960)
    cap.set(4,1080)
    cv2.namedWindow("frame")#,cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 1920, 1080)
   # cv2.resizeWindow("frame", 600, 400)
    cv2.moveWindow("frame", 0,0)
    no_play=True
    ret=True
    genderRatio=0
    timeToChange=0
    previousCom=-1
    aLanguage=False
    translate = translator() 
    
    while ret:
        ret, or_img=cap.read()
        
        img = np.zeros((len(or_img),len(or_img[1])*2,3), np.uint8)
        cu=time.time()-now
        if not ret:
            break
        bg_img = cv2.cvtColor(or_img, cv2.COLOR_BGR2GRAY)
        del_list[:] = []
        faceImgList=[]
        for idx, face in enumerate(faces):
            try:
                current=face.pts
                tr_img*=0
                bg_img = cv2.cvtColor(or_img, cv2.COLOR_BGR2GRAY)
                dd=current[21, :]-current[12, :]
                angle=np.angle(np.complex(dd[0],dd[1]))/np.pi*180
                rows, cols=bg_img.shape
                M=cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                bg_img=cv2.warpAffine(bg_img, M, (cols, rows))
                new_pt=np.array(np.matrix(M)*np.vstack([current.T, np.ones((1, n_point))])).T
                mmX, mmY, norm_D=norm_dis(tracker_config, new_pt.copy())
    #            
    #            mmX, mmY, norm_D=norm_dis(tracker_config, current.copy())
                bbs=np.round(create_box(tracker_config, mmX, mmY, norm_D)).astype('int')
                current_T=tracker_average.copy()
                scale=tracker_config['img_shape']/float(bbs[2]-bbs[0])
                box_left=max(bbs[0], 0)
                box_right=min(bbs[2], or_img.shape[1]-1)
                box_up=max(bbs[1], 0)
                box_down=min(bbs[3], or_img.shape[0]-1)
                st_le=tracker_pad+int(round((box_left-bbs[0])*scale))
                st_ri=int(round((box_right-box_left)*scale))
                st_up=tracker_pad+int(round((box_up-bbs[1])*scale))
                st_do=int(round((box_down-box_up)*scale))
            #        print  bbs, box_left, box_right, box_up, box_down
                tr_img[st_up:st_up+st_do, st_le:st_le+st_ri]=cv2.resize(bg_img[box_up:box_down,box_left:box_right],(st_ri,st_do), interpolation = cv2.INTER_CUBIC)
                #fillin image,generate start_point, and scale
            #        print tr_img.shape
                
                for ii in range(tr_iter):
                    tr_feature=tracker_calc_hog(tracker_config, tracker_pad, current_T.copy(), ii, tr_img, tr_hog[ii])
                    if ii==tr_iter-1:
                        assert classifier_model.predict_proba(tr_feature.reshape((1,-1)))[0,1]>0.05
        #                    print name, ' failed'
                    if ii==0: hidden=get_activations(tr_feature)
                    if ii!=0:residue, current_T, trs=procrustes(tracker_average, current_T)
                    delta_x=tracker_models[ii].predict(tr_feature.reshape((1,-1)))
                    current_T+=delta_x.reshape((-1,2), order='F')
                    if ii!=0:current_T=inv_procrustes(current_T, trs)
                faceImgList.append(tr_img[tracker_pad:-tracker_pad, tracker_pad:-tracker_pad].copy())
                ftr=np.concatenate([tr_feature.ravel(), hidden.ravel()]).reshape((1, -1))
                current=reverse_point(current_T, bbs[0:2].copy(),scale)
                current=np.dot(M[:, 0:2].T, (current.copy()-np.vstack([M[:,2]]*36)).T).T
                face.pts=current
                face.on_mode(agemodel.predict_proba(ftr), gendermodel.predict_proba(ftr), racemodel.predict_proba(ftr), emotionmodel.predict_proba(ftr),avimodel.predict(ftr))
                del_list.append(face)
            except:
                pass
                #print "EXCEPTION RAISED 1"
        faces=del_list[:]

            
           
        if cu>thresh or len(faces)==0: 
            if cu<=thresh: 
                thresh=cu
            else:
                thresh+=1
            bbxs=faceCrop(or_img, faces, (1, 1.2))
#            if not bbxs:
#                cv2.imshow('frame', or_img)
#                aa=cv2.waitKey(1)
#                if aa & 0xFF == ord('q'):
#                    break
#                continue
            for bbs in bbxs:
                if bbs[-1]<=bg_img.shape[0]:
                    img=bg_img[bbs[1]:bbs[3], bbs[0]:bbs[2]]
                else:
                    img=bg_img[bbs[1]:bg_img.shape[0], bbs[0]:bbs[2]]
                    img=np.lib.pad(img, ((0, bbs[3]-bg_img.shape[0]), (0,0)), 'constant', constant_values=((0,0),(0,0)))
                bbs=np.array(bbs)
                bbs[2:]-=bbs[:2]
                start_point, scale=point_calc(config, bbs.copy())# transform to cv2 face crop space(FCS)
                img=img[(start_point[1]-bbs[1]):, :]
                tr_img*=0
                current_T=average.copy()
                xsize=int(tracker_config['img_shape'])
                tr_img[tracker_pad:tracker_pad+xsize, tracker_pad:tracker_pad+xsize]=cv2.resize(img,(xsize,xsize), interpolation = cv2.INTER_CUBIC)
                for ii in range(de_iter): 
        #                calc_hog(config, current_T, ii, img, feature, block_feature,de_hog[ii])            
                    feature=tracker_calc_hog(config, tracker_pad, current_T.copy(), ii, tr_img, de_hog[ii])
                    residue, current_T, trs=procrustes(average, current_T.reshape((n_point,2), order='F'))
                    delta_x=models[ii].predict(feature.reshape((1,-1)))
                    current_T+=delta_x.reshape((-1,2), order='F')
                    current_T=inv_procrustes(current_T, trs)
                current=reverse_point(current_T, start_point.copy(), scale)
                faces.append(face_tmp(current, id_dis))
                pool_dict[id_dis]=collect_tmp()
                id_dis+=1
        
#            current=np.dot(M[:, 0:2].T, (current.copy()-np.vstack([M[:,2]]*36)).T).T
#        print len(faces)
        if no_play and cu>5: 
#            subprocess.Popen('"C:\\Program Files\\VideoLAN\\VLC\\vlc.exe" 24.mp4')
            no_play=False
        #save_flag=1
        save_flag=False
        for face in faces:
            if face.mode==1:
                bbox=np.array([np.min(face.pts[pt_sel, :], axis=0), np.max(face.pts[pt_sel, :], axis=0)]).ravel().astype('int')
                face.green.append(np.mean(or_img[bbox[1]:bbox[3], bbox[0]:bbox[2], 1]))
                face.time.append(cu)
                idx=pool_dict[face.id]
                idx.on_mode(face)
                if cu-face.time[0]>20 and cu-face.last>1:
                    try:
                        freq,feature=HR_feature(np.array(face.time),np.array(face.green))
                        hhr=freq[18+np.argmax(feature[0,18:])]
                        if hhr>50:
                            idx.hr.append(hhr)
#                            idx.hr.append(hrmodel.predict(feature))
                        idx.time.append(cu)
                        face.last=cu
                        
#                        cv2.putText(or_img, hr, (int(face.pts[21, 0]+50), int(face.pts[12, 1]+30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                    except:
                        pass
                        #print "exception raised 2"
                #cv2.rectangle(or_img, (int(face.pts[21, 0]+50), int(face.pts[21, 1])-20), (int(face.pts[21, 0]+50+xs[0][0]), int(face.pts[21, 1])-20+xs[0][1]*5), (0,0,0), -1)
                if idx.hr:
                    cv2.putText(or_img, '%d'%idx.hr[-1], (int(face.pts[21, 0]+50), int(face.pts[12, 1]+30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                try:
                    attn=estimate_projection(face.pts[[6,8,12,15,18,21,24,30], :], td_mean, 1080)
                    
                    idx.update_attn(attn)
                    face.attn=attn
                    #cv2.putText(or_img, '%d%%'%(attn*100), (int(face.pts[21, 0]+50), int(face.pts[12, 1]+55)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                except:
                    pass
                    #print "EXCEPTION RAISED 3"
                #ARYEL
                face.idx=idx
                for ii in [6,8,12,15,18,21,24,30]:#range(current.shape[0]): 
                    cv2.circle(or_img, tuple(face.pts[ii, :].astype('int')), 2, color=(0, 0,255), thickness=1)
                
                if idx.gender=="M":
                    face.genderS="Male"
                else:
                    genderRatio+=1
                    face.genderS="Female"
                
                face.raceS=idx.race
                face.ageS=idx.age
                face.emo_label=idx.emo_label
                face.avi_label=[float(idx.avi_label[0]),float(idx.avi_label[1])]
                #print face.avi_label
                #cv2.rectangle(or_img, (int(face.pts[12, 0]-50-xs[0][0]), int(face.pts[12, 1])-20-xs[0][1]), (int(face.pts[12, 0]-50), int(face.pts[12, 1])-20+xs[0][1]*4), (0,0,0), -1)
                #cv2.putText(or_img, idx.gender, (int(face.pts[12, 0]-50-xs[0][0]), int(face.pts[12, 1])-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                #cv2.putText(or_img, idx.race, (int(face.pts[12, 0]-50-xs[0][0]), int(face.pts[12, 1]+5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                #cv2.putText(or_img, idx.age, (int(face.pts[12, 0]-50-xs[0][0]), int(face.pts[12, 1]+30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                #cv2.putText(or_img, idx.emo_label, (int(face.pts[12, 0]-50-xs[0][0]), int(face.pts[12, 1]+55)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
               # save_flag&=(idx.emo_label!='NE')
                
                #cv2.putText(or_img, '%.2f'%idx.avi_label[0], (int(face.pts[21, 0]+50), int(face.pts[12, 1]-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                #cv2.putText(or_img, '%.2f'%idx.avi_label[1], (int(face.pts[21, 0]+50), int(face.pts[12, 1]+5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
        

        new_img, or_img = GroupSensing(faces, translate, faceImgList, or_img, aLanguage)
        #vis = np.concatenate((or_img, new_img), axis=1)
        
        cv2.imshow('frame', or_img)
        cv2.resizeWindow("frame", 960, 540)
        cv2.imshow('frame1', new_img)
        
        
        aa=cv2.waitKey(1)
        if timeToChange<200: 
            timeToChange+=1
            
        if len(faces)>0 and timeToChange>50: # 0 = female # 1 =male # 2 =FemaleG #3=MaleG TO DO use dic
            com=-1
            if genderRatio==len(faces):
                if len(faces)>1:
                    commercial=displayCommercial(2)
                    com=2
                else:
                    commercial=displayCommercial(0)
                    com=0
            elif genderRatio==0:
                if len(faces)>1:
                    commercial=displayCommercial(3)
                    com=3
                else:
                    commercial=displayCommercial(1)
                    com=1
            else:
                commercial=displayCommercial(4)
                com=4

            genderRatio=0
            if com!=previousCom:   
                #cv2.namedWindow("frame2",cv2.WINDOW_NORMAL)
                cv2.imshow('frame2', commercial)
                #cv2.moveWindow('frame2',700,400)
                #cv2.resizeWindow("frame", 50, 50)
                timeToChange=0
                previousCom=com
                
        
        if aa & 0xFF == ord('q'):
            break
        if aa & 0xFF == ord('l'):
            aLanguage=not aLanguage

        
        #if len(faces)>0 and save_flag:
         #   cv2.putText(or_img, '%.2f'%cu, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
          #  input_queue.put(or_img)
        
    cap.release()
    cv2.destroyAllWindows()
    print time.time()-now
    input_queue.put(np.zeros((10,10)))
    while True:
        try:    
            input_queue.get_nowait()
            input_queue.task_done()
        except:
            print "EXCEPTION RAISED 4"
            break
    input_queue.close()
    input_queue.join()
   # visualize(pool_dict, cu)
#