# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 08:31:05 2020

@author: Lenovo T420s
"""
import spacy
import numpy as np
from numpy import dot
from numpy.linalg import norm
from spacy.matcher import Matcher
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


import buildSenten as build


nlp = spacy.load("en_core_web_lg") 

def similarity1(v1,v2):
    return (dot(v1, v2)/(norm(v1)*norm(v2)))

def Manhattan(v1,v2):
    return sum(abs(v2-v1))

def preparationTokenSpCy(totalSpeachGr):
    def  traitment1(a):
        return (a[0],a[1],a[2], nlp(a[3]))
    return list(map(traitment1,totalSpeachGr ))  



def sentCount(doc)-> int: 
    return(sum(1 for _ in doc.sents))

def lemmeCount(doc)->dict:
    res = { }
    for elem in doc:
       if elem.lemma_ in res.keys():
           res[elem.lemma_] += 1
       else:
           res[elem.lemma_] = 1
    return res

def df(lemme : str, lemmeCount :dict, norm = True, n = 1) -> float:
    r = lemmeCount.get(lemme, 0)
    if norm:
        return r + n
    else:
        return r

def dfSent(lemmes :list, lemmeCount : dict, norm = True, n = 1) -> list:
    def dfSimple1(lemme):
        return lemmeCount.get(lemme, 0)
    def dfSimple2(lemme):
        return lemmeCount.get(lemme, 0) + n
    
    if norm:
        return list(map(dfSimple2, lemmes))
    else :
        return list(map(dfSimple1, lemmes))
    
    return []

def lemmeCounterFusion(lemmeCounters: list) -> dict:
    l = []
    for lc in lemmeCounters:
        l.extend(list(lc.keys()))
    l = list(set(l))
    
    res = { }
    for k in l:
        r = 0
        for lc in lemmeCounters:
            r += lc.get(k,0)
        res[k]= r
    return res


def typeOfWordFCount( doc)->dict:
    res = { }
    for elem in doc:
       if elem.pos_ in res.keys():
           res[elem.pos_] += 1
       else:
           res[elem.pos_] = 1
    return res 


def listOfnameEntity(doc)-> list:
    """TODO
    res = { }
    for elem in doc:
       if elem.pos_ in res.keys():
           res[elem.pos_] += 1
       else:
           res[elem.pos_] = 1
           """
    return []



def extractionEntiteNome(peoples:list,speach: list)-> list:
    res = []
    for elem in speach:
        if elem[0] in peoples:
            res.append(typeOfWordFCount(elem[3]))
    return lemmeCounterFusion(res)

    return []

""" interotion count"""
def interCount(doc)->int:
    cpt = 0 
    for sent in doc.sents:
        if sent[-1].text == '?':
            cpt += 1
    return cpt

def interogationCounter(peoples:list,speach:list )-> list:
    res = []
    for elem in speach:
        if elem[0] in peoples:
            i = interCount(elem[3])
            if i > 0:
                res.append((elem[0], elem[1],i))
    return res
    
def regroupSimple(l:list, window = 120 ):
    res = { }
    for e in l:
        pos = window * round(   e[1] / window)
        res[pos] = res.get(pos,0) + e[2]
        
    return sorted(res.items(), key=lambda t: t[0])

""" Named Entities extraction"""
def docToNamesEntities(peoples:list,speach:list)-> list:
    res = []
    for elem in speach:
        if elem[0] in peoples:
            res.extend( list(map(lambda x : (elem[0],elem[1],x) ,elem[3].ents)))
    return res

def mmFilter(enList:list)->list:
    return list(filter(lambda x: (not(' mm'in x[2].text)  and not( 'mm'== x[2].text.lower())) , enList))
    
def filterNamesEntities(enList:list,types:list)->list:
    return list(filter(lambda x: x[2].label_ in types , enList))

 

def speakerName(peoples:list,speach:list)-> list:
    listOfDoc = []
    for elem in speach:
        if elem[0] in peoples:
            listOfDoc.append(elem[3])
    res = []
    filt = [{"POS": "PROPN","OP": "+" }, {"IS_PUNCT": True }, {"LOWER": "you" }]
    matcher = Matcher(nlp.vocab)
    matcher.add("speaker",None,filt)
    for d in listOfDoc:
        matches = matcher(d)
        for match_id, start, end in matches:
            res.append((d[start],d[start:end]))                        
    return res



 
    
    
""" verb features """
    
def docToVerb(peoples:list,speach:list)-> list:
    res = []
    for elem in speach:
        if elem[0] in peoples:
            for word in elem[3]:
                if word.dep_ in ['ROOT','ccomp'] and word.pos_ == 'VERB':
                    res.append((word.lemma_,word.vector))
                
    return res 



def freqtypeExtractor(peoples:list , speach: list )-> dict:

    res = []
    for elem in speach:
        if elem[0] in peoples:
            res.append(typeOfWordFCount(elem[3]))
    return lemmeCounterFusion(res)



"""
parameter 
    people :list of people (str)
    speach : list of sent (people,begin, end, sentence parse with sapcy)
return :the lemme frequence for a groupe of peoples in the speach
"""
    
def freqExtractor(peoples:list , speach: list )-> dict:

    res = []
    for elem in speach:
        if elem[0] in peoples:
            res.append(lemmeCount(elem[3]))
    return lemmeCounterFusion(res)

def docSpacySentPretraitement(subMeetings,meeting = ""):
    res = { }
    for sm in subMeetings:
        try:
            
            if meeting == "":
                orgSpeachT = build.organisedSpeachTotalPipe(sm)
            else:
                orgSpeachT = build.organisedSpeachTotalPipe(sm, meeting)
            res[sm] = { }
            res[sm]["meeting"] = preparationTokenSpCy(orgSpeachT)
        except OSError:
            print("erreur lors du chargement de " +meeting+ " " + sm )
        except TypeError:
            print("erreur lors du chargement de " +meeting+ " " + sm )
    return res

def addFreqToDSS(peoples,dss):
    for m in dss.keys():
        dss[m]["freq"] = { }  
        dss[m]["typeFreq"] = { }
        for i in peoples[m]:
            dss[m]["freq"][i] = freqExtractor([i],dss[m]["meeting"])
            dss[m]["typeFreq"][i] = freqtypeExtractor([i],dss[m]["meeting"])
    return dss

def normalisation(vect):
    s = sum(vect)
    def  traitment1(a):
        return a/s
    
    return list(map(traitment1,vect ))

def typeVectorComputation(typeOfword):
    lType= ['PUNCT','VERB','CCONJ','X','ADV','NUM','PRON','ADP','INTJ','PART',
            'NOUN', 'DET', 'ADJ', 'PROPN']
    def  traitment1(a):
        return (typeOfword.get(a,0))
    v =  list(map(traitment1,lType ))
    res =     normalisation(v)
    return np.array(res)

def addfreqTypeVectorToDSS(peoples,dss):
    for m in dss.keys(): 
        dss[m]["typeVector"] = { }
        for i in peoples[m]:
            dss[m]["typeVector"][i] = typeVectorComputation(dss[m]['typeFreq'][i])
    return dss

def getBestwordsDSS(dss):
    res = []
    for m in dss.keys():
        resp = []
        for p in dss[m]["freq"].keys():
            r = dss[m]["freq"][p]
            sorted(r, key=r.__getitem__,reverse=True)
            resp.append( (p,sorted(r, key=r.__getitem__)[:10])) 
        res.append((m,resp))
    return res

def getBestTypewordsDSS(dss):
    res = []
    for m in dss.keys():
        resp = []
        for p in dss[m]["typeFreq"].keys():
            r = dss[m]["typeFreq"][p]
            sorted(r, key=r.__getitem__,reverse=True)
            resp.append( (p,sorted(r, key=r.__getitem__)[:5])) 
        res.append((m,resp))
    return res

def statFreqDSS(dss):
    res = []
    for m in dss.keys():
        resp = []
        cpt = 0
        for p in dss[m]["freq"].keys():
            resp.append((p,sum(dss[m]["freq"][p].values())))
            cpt += resp[-1][1]        
        res.append((m,resp,cpt))
    return res

"""Verb"""
def addVerbToDSS(peoples,dss):
    for m in dss.keys(): 
        dss[m]["verb"] = { }
        for i in peoples[m]:
            dss[m]["verb"][i] = docToVerb([i], dss[m]["meeting"])
    return dss


def addVerbVectorToDSS(peoples,dss):
    for m in dss.keys(): 
        dss[m]["verbVector"] = { }
        for i in peoples[m]:
            dss[m]["verbVector"][i] = verbVectorComputation(dss[m]['verb'][i])
    return dss
    
def verbVectorComputation(verbs):
    def  traitment1(a):
        return (a[1])
    return np.average( np.array( list(map(traitment1,verbs )) ),0)

def peopleVectorExtraction(peoples,dss,vectorName):
    res = []
    for m in dss.keys(): 
        for i in peoples[m]:
            res.append((str(m) + str (i),dss[m][vectorName][i]))
    return res
 
def similarityPeople(pv):
    res = []
    for pos in range (0, len(pv)):
        for pos2 in range (pos + 1,len(pv)):
            res.append((pv[pos][0],pv[pos2][0], similarity1(pv[pos][1],pv[pos2][1])))
    return res

def similarityPeopleClassic(pv):
    res = []
    for pos in range (0, len(pv)):
        for pos2 in range (pos + 1,len(pv)):
            res.append((pv[pos][0],pv[pos2][0], Manhattan(pv[pos][1],pv[pos2][1])))
    return res                                


"""apprentisage debut fin"""
def dssToTrainCorpusDebutFin(dssList,size):
    part = []
    x1= []
    x2=[]
    y= []
    """extraction des parties"""
    for v0 in dssList:
        for k1, v1 in v0.items():
            c = { }
            for k in v1['meeting']:
                space = c.get(int(k[1]/size),[])
                space.append(k)
                c[int(k[1]/size)]= space
            deb = min(c.keys())
            fin = max(c.keys())
            for k3,v3 in c.items():
                if deb == k3:
                    part.append((1,v3))
                elif fin == k3:
                    part.append((2,v3))
                else:
                    part.append((0,v3))
    
    for p in part:
        vect1 = typeVectorComputation( freqtypeExtractor(['A','B','C','D'],p[1]))
        vect2 = verbVectorComputation(docToVerb(['A','B','C','D'],p[1]))
        x1.append(vect1)
        x2.append(vect2)
        y.append(p[0])
         
    
    return x1,x2,y


def multipleDSS():
    meetings = ['a','b','c','d']
    submeet = ["IS1000","IS1001","IS1002","IS1003","IS1004","IS1005","IS1006","IS1007","IS1008","IS1009"]
    dssl= []
    for i in submeet:
        print(i)
        dssl.append(docSpacySentPretraitement(meetings,i))
    return dssl

def testFin():
    d = multipleDSS()
    x1,x2,y = dssToTrainCorpusDebutFin(d,5*60)
    print("type freq")
    X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    accuracy_score(y_test.values, y_predict)
    
    print("verb")
    X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    accuracy_score(y_test.values, y_predict)

                    
    
    
testFin()    

""" 
    
print("begin", 0.0)   

meetings = ['a','b','c','d']
dss = docSpacySentPretraitement(meetings,"IS1000")
peoples =  { }
for m in meetings:
        peoples[m] = ['A','B','C','D']

dss = addFreqToDSS(peoples,dss)
dss = addVerbToDSS(peoples,dss)
dss = addVerbVectorToDSS(peoples,dss)
addfreqTypeVectorToDSS(peoples,dss)


r = peopleVectorExtraction(peoples,dss,'typeVector')
s= similarityPeopleClassic(r)
print(s)
"""

"""
print(getBestwordsDSS(dss))
print(getBestTypewordsDSS(dss))
r = statFreqDSS(dss)
print(r)

"
l = ['A','B','C','D']
print(regroupSimple( interogationCounter(l, dss['b']["meeting"])))
en = docToNamesEntities(l, dss['b']["meeting"])
print('date',filterNamesEntities(en,['DATE']))
print('localisation and org',filterNamesEntities(en,['GPE','ORG']))
print('person ',filterNamesEntities(en,['PERSON']))
print('person ',mmFilter( filterNamesEntities(en,['PERSON'])))

print(speakerName(l, dss['b']["meeting"]))
vb = docToVerb(l, dss['b']["meeting"])
for elem in vb:
    print(elem[0])

for e in ['a','b','c','d']:
    print(e)
    t = time.time()
    orgSpeachT = build.organisedSpeachTotalPipe(e)
    speachWspacy = preparationTokenSpCy(orgSpeachT)
    print("spacy traitement done", time.time() - t)
    print("text extractor", time.time() - t)
    for i in ['A','B','C','D']:
        r = freqExtractor([i],speachWspacy)
        sorted(r, key=r.__getitem__,reverse=True)
        print(i,sorted(r, key=r.__getitem__)[:10])
"""


