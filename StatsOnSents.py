# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 08:31:05 2020

@author: Lenovo T420s
"""

import math
import spacy
import numpy as np
from numpy import dot
from numpy.linalg import norm
from spacy.matcher import Matcher
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


import buildSenten as build
import peopleType as meetingRoles

nlp = spacy.load("en_core_web_lg") 
peopleInMeeting = ['A','B','C','D']
submeet = ["IS1000","IS1001","IS1002","IS1003","IS1004","IS1005",
           "IS1006","IS1007","IS1008","IS1009"]

submeet2 = ["IS1000","IS1001","IS1002","IS1003","IS1004","IS1005",
           "IS1006","IS1007","IS1008","IS1009",
           "ES2002","ES2003","ES2004","ES2005",
           "ES2006","ES2007","ES2008","ES2009", "ES2010","ES2011"]

def similarity1(v1,v2):
    return (dot(v1, v2)/(norm(v1)*norm(v2)))

def Manhattan(v1,v2):
    return sum(abs(v2-v1))


def meetingTime(meeting,begin,end):
    def ftemps(x):
        return (x[1]>begin and x[2]<end)
    return filter(ftemps,meeting)


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

def extractionEntiteNome(peoples:list,speach: list)-> list:
    res = []
    for elem in speach:
        if elem[0] in peoples:
            res.append(typeOfWordFCount(elem[3]))
    return lemmeCounterFusion(res)

    return []


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
    verbs = []
    for elem in speach:
        if elem[0] in peoples:
            for word in elem[3]:
                if word.dep_ in ['ROOT','ccomp'] and word.pos_ == 'VERB':
                    verbs.append((word.lemma_,word.vector))
                
    verb = { }
    for v in verbs:
        if v[0] in verb.keys():
            verb[v[0]][0] +=1
        else:
            verb[v[0]]=[1,v[1]]
    return verb


"""Verb"""
    
    


def addPeoplesVerbToDSS(peoples,dss):
    for m in dss.keys(): 
        dss[m]["verb"] = { }
        for i in peoples[m]:
            r = docToVerb([i], dss[m]["meeting"])
            dss[m]["verb"][i] = r                 
    return dss

def getVerbIDFCorpus(dss):
    l = { }
    totaldoc = 0
    for m in dss.values():
        verbs = m["verb"]
        for lv in verbs.values():
            totaldoc += 1
            for kv,vVal in lv.items():
                l[kv] = l.get(kv,0.0) + 1.0
    for k in l.keys():
        l[k] = math.log( totaldoc /  l[k])
    return l

        

def addVerbVectorToDSS(peoples,dss):
    idf = getVerbIDFCorpus(dss)
    for m in dss.keys(): 
        dss[m]["verbVector"] = { }
        for i in peoples[m]:
            dss[m]["verbVector"][i] = verbVectorComputationTFIDF(dss[m]['verb'][i],idf)
    return dss
    
def verbVectorComputation(verbs):
    def  traitment1(a):
        return (verbs[a][1])
    v = list(verbs.keys())
    return np.average( np.array( list(map(traitment1,v )) ),0)


def verbVectorComputationTFIDF(verbs,idf):
    def  vectPond(a):
        return ((verbs[a][0] * idf[a]) *verbs[a][1]  )
    def tfidf(a):
        return (verbs[a][0] * idf[a])
    vlist = list(verbs.keys())
    s = sum(map(tfidf,vlist))
    av = np.average( np.array( list(map(vectPond,vlist )) ),0)
    return av / s






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




def addFreqToDSS(peoples,dss):
    for m in dss.keys():
        dss[m]["freq"] = { }  
        dss[m]["typeFreq"] = { }
        for i in peoples[m]:
            dss[m]["freq"][i] = freqExtractor([i],dss[m]["meeting"])
            dss[m]["typeFreq"][i] = freqtypeExtractor([i],dss[m]["meeting"])
    return dss

def addDictionaries(dss ):
    for m in dss.keys():
        dss[m]["dictionary"] = freqExtractor(peopleInMeeting,dss[m]["meeting"])
    return dss 

def getTFIDFGlobal(dss):
    l = { }
    totaldoc = 0
    for m in dss.values():
        dic = m["dictionary"]
        totaldoc += 1
        for kv,vVal in dic.items():
                l[kv] = l.get(kv,0.0) + 1.0
    for k in l.keys():
        l[k] = math.log( totaldoc /  l[k])
    return l

def getCorpusDictionary(dss):
     
    l = []
    for m in dss.values():
        l.append(m["dictionary"])
    return lemmeCounterFusion(l)
    
    




    
        


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

"""Vector"""



def getMeetingTypeVector(Meeting,  begin = 0, end = 60 * 45):
    
    temp = meetingTime( Meeting,begin,end )
    return typeVectorComputation(freqtypeExtractor(peopleInMeeting,temp))

def getAllMeetingsTypeVector(dss):
    l = []
    for i in dss.keys():
        l.append((i ,getMeetingTypeVector(dss[i]['meeting']) ))
    
    return l
        
    
    
    
    

def getMeetingVerbVectors(idfVerb, Meeting , begin = 0, end = 60 * 45):
    temp = meetingTime( Meeting,begin,end )
    verbs = docToVerb(peopleInMeeting,temp)

    return verbVectorComputationTFIDF(verbs,idfVerb)

def getAllMeetingsVerbVector(dss):
    idf = getVerbIDFCorpus(dss)
    l = []
    for i in dss.keys():
        l.append((i ,getMeetingVerbVectors(idf,dss[i]['meeting']) ))
    return l
    



def peopleVectorExtraction(peoples,dss,vectorName):
    res = []
    for m in dss.keys(): 
        for i in peoples[m]:
            res.append((str(m+i),dss[m][vectorName][i]))
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


""" Apprentissage"""


def clustering(pv):
    vect = []
    p = []
    for i in pv:
        vect.append(i[1])
        p.append(i[0])
    k = KMeans().fit(np.array(vect))
    lab = k.labels_
    for i in range (0,7):
        print(i)
        for j in range(0,len(p)):
            if (lab[j] == i):
                print(p[j])





"""apprentisage debut fin"""


def dssToTrainCorpusDebutFin2(dss,size):
    part = []
    x1= []
    x2=[]
    y= []
    """extraction des parties"""
    for k1, v1 in dss.items():
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

def dssToTrainCorpusRole(dss):
    def lToDict(l):
        d = { }
        for e in l:
            d[e[0]]= e[1]
        return d
    
    role = meetingRoles.AllRoles()
    pp = peoplesgenerator(dss)
    pVV = lToDict(peopleVectorExtraction(pp,dss,'verbVector'))
    pTV = lToDict(peopleVectorExtraction(pp,dss,'typeVector'))
    
    
    x1= []
    x2=[]
    y= []

    
    for p in pVV.keys():
        x1.append(pTV[p])
        x2.append(pVV[p])
        y.append(role[p])
                  
    
    return x1,x2,y





def testFin(dss):
    x1,x2,y = dssToTrainCorpusDebutFin2(dss,5*60)
    print("type freq")
    X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(class_weight={1:0.15,2:0.15,0:0.7})
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    print(accuracy_score(y_test, y_predict))
    st = accuracy_score(y_test, y_predict)
    
    print("verb")
    X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(class_weight={1:0.15,2:0.15,0:0.7})
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    print(accuracy_score(y_test, y_predict))
    return(st,accuracy_score(y_test, y_predict))
    
def testRole(dss):
    x1,x2,y = dssToTrainCorpusRole(dss)
    
    print("type freq")
    X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    print(accuracy_score(y_test, y_predict))
    st = accuracy_score(y_test, y_predict)
    
    print("verb")
    X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    print(accuracy_score(y_test, y_predict))
    return(st,accuracy_score(y_test, y_predict))



def multipleTimeTest(dss, nb = 10):
    roleVerb = []
    roleType = []
    begVerb = []
    begType = []
    for i in range (0,nb):
        t,v = testRole(dss)
        roleType.append(t)
        roleVerb.append(v)
        
        t,v = testFin(dss)
        begType.append(t)
        begVerb.append(v)
    return(roleType,roleVerb,begType,begVerb) 
        
        
    
    
    
    
    
"""Corpus Generator"""

def preparationTokenSpCy(totalSpeachGr):
    def  traitment1(a):
        return (a[0],a[1],a[2], nlp(a[3]))
    return list(map(traitment1,totalSpeachGr ))  

"""old, only for one meeting, prefer multipleDSS in any case"""
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

def AddNewMeeting(dss,meeting,subMeetings):
    if meeting in (dss.keys()):
        print("allready in",meeting  )
    else:
        for subMeeting in subMeetings:
            name = meeting + subMeeting
            try:
                orgSpeachT = build.organisedSpeachTotalPipe(subMeeting, meeting)
                dss[name] = { }
                dss[name]["meeting"] = preparationTokenSpCy(orgSpeachT)
            except OSError:
                print("erreur lors du chargement de " + name )
            except TypeError:
                print("erreur lors du chargement de " + name )
    return dss

def  multipleDSS(subMeetings = ['a','b','c','d'],meetings = ["IS1000","IS1001","IS1002"]):
    dss = { }
        
    
    for i in meetings:
        print(i)
        AddNewMeeting(dss, i,subMeetings)
    return dss

def peoplesgenerator(dss):
    l = { }
    for i in dss.keys():
        l[i] = peopleInMeeting
    return l


def AddAll(dss):
    pp = peoplesgenerator(dss)
    addFreqToDSS(pp,dss)
    addfreqTypeVectorToDSS(pp,dss)
    addPeoplesVerbToDSS(pp,dss)
    addVerbVectorToDSS(pp,dss)
    addDictionaries(dss)    
    
    
def moncorpustest(meets):
    d = multipleDSS(['a','b','c','d'],meets)
    AddAll(d)
    return( d)

    
    


                    
"""    
    
testFin()    

 
    
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


