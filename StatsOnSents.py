# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 08:31:05 2020

@author: Lenovo T420s
"""
import spacy
import buildSenten as build
import time

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

def filterNamesEntities(enList:list,types:list)->list:
    return list(filter(lambda x: x[2].label_ in types , enList))
    
    
    
    

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

def docSpacySentPretraitement(subMeetings):
    res = { }
    for sm in subMeetings:
        res[sm] = { }
        orgSpeachT = build.organisedSpeachTotalPipe(sm)
        res[sm]["meeting"] = preparationTokenSpCy(orgSpeachT)
    return res

def addFreqToDSS(peoples,dss):
    for m in dss.keys():
        dss[m]["freq"] = { }  
        dss[m]["typeFreq"] = { }
        for i in peoples[m]:
            dss[m]["freq"][i] = freqExtractor([i],dss[m]["meeting"])
            dss[m]["typeFreq"][i] = freqtypeExtractor([i],dss[m]["meeting"])
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

    

print("begin", 0.0)   
t = time.time()   
nlp = spacy.load("en_core_web_lg") 
print("nlp done", time.time() -t)

meetings = ['b']
dss = docSpacySentPretraitement(meetings)
peoples =  { }
for m in meetings:
        peoples[m] = ['A','B','C','D']
"""
dss = addFreqToDSS(peoples,dss)
print(getBestwordsDSS(dss))
print(getBestTypewordsDSS(dss))
r = statFreqDSS(dss)
print(r)

"""
l = ['A','B','C','D']
print(regroupSimple( interogationCounter(l, dss['b']["meeting"])))
en = docToNamesEntities(l, dss['b']["meeting"])
print('date',filterNamesEntities(en,['DATE']))
print('localisation and org',filterNamesEntities(en,['GPE','ORG']))
print('person ',filterNamesEntities(en,['PERSON']))

"""
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


