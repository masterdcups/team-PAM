# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 08:37:35 2020

@author: Lenovo T420s
"""

import buildSenten as build
import matplotlib.pyplot as plt
import numpy as np


def preparationTimeSpeak(totalSpeachGr):
    def  traitment1(a):
        return (a[0],a[1], a[2],)
    return list(map(traitment1,totalSpeachGr ))


def pretraitementOne(subMeetings,meeting = "" ):
    res = { }
    for sm in subMeetings:
        try:
            if meeting == "":
                 
                orgSpeachT = build.organisedSpeachTotalPipe(sm)
            else:
                orgSpeachT = build.organisedSpeachTotalPipe(sm, meeting)
            res[meeting + sm] = { }
            res[meeting + sm]["timeSent"] = preparationTimeSpeak(orgSpeachT)
        except OSError:
            print("erreur lors du chargement de " +meeting+ " " + sm )
        except TypeError:
            print("erreur lors du chargement de " +meeting+ " " + sm )
            
    return res

def pretraitement(meetings):
    c = { }
    for m in meetings:
        c.update(pretraitementOne(m[1],m[0]))
    return c
    
def ISMeeting():
    ism = "IS100"
    number = 10
    part = ["a","b","c","d"]
    res = []
    for i in range (0, number):
        res.append((ism + str(i), part))
    return res

def tSMeeting():
    tsm = "TS300"
    tsm10 = "TS30"
    number = 13
    part = ["a","b","c","d"]
    res = []
    for i in range (3, number):
        if i < 10:
            res.append((tsm + str(i), part))
        else:
            res.append((tsm10 + str(i), part))
    return res

"""ajout du temps par personne"""
def timeBySpeaker(TSMeeting):
    res = { }
    for t in TSMeeting:
        res[t[0]] = res.get(t[0], 0) + t[2] - t[1]
    return res

def AddTimeBySpeaker(corpus):
    for k in corpus.keys():
        corpus[k]["timeSpk"]= timeBySpeaker(corpus[k]["timeSent"])



def bestSpk(TSpkMeeting):
    first = True
    tot = 0
    for k,v in TSpkMeeting.items():
        tot += v
        if first :
            res= (k,v)
            first = False
        elif v > res[1]:
            res = (k,v)
    if not first:
        return (res[0], res[1],res[1] / tot)
    else: return (None,0,0)

def AddBestSpeaker(corpus):
    for k in corpus.keys():
        corpus[k]["BestSpk"]= bestSpk(corpus[k]["timeSpk"])


def bestSpkRepartition(corpus, TimeProportionlength):
    res = { }
    for v in corpus.values():
        n = int (v["BestSpk"][2] / TimeProportionlength)
        res[n * TimeProportionlength] = res.get(n * TimeProportionlength,0) + 1
    return sorted(res.items())


def affichageSPKRep(result):
    plt.plot(np.array(result[0]) , np.array(result[1]), label='repartition')
    plt.xlabel('proportionSpeak')
    plt.xlabel('number')
    plt.title("Repartition de la proportionde parole du plus grand parleur au sein des réunions ")
    plt.legend()
    plt.show()

def AddRole(corpus,timeprop,ecart,sc1,sc2):
    for v in corpus.values():
        if v["BestSpk"][2] > timeprop:
            b = bestBySection(v,ecart)
            score = countSection(v["BestSpk"][0],b)
            if score > sc2 :
                v["BestSpk"].append("coordinator")
            elif score >sc1:
                v["BestSpk"].append("presenter")
                

def timeBySpeakerSection(TSMeeting,beg,en):
    res = { }
    for t in TSMeeting:
        if t[1]<beg and t[1]> en:
            res[t[0]] = res.get(t[0], 0) + t[2] - t[1]
    return res


def bestBySection(meeting, ecart):
    r = []
    for  i in range(0,2700, ecart):
        tmp = bestSpk(timeBySpeakerSection(meeting["timeSent"],i,i + ecart))
        if tmp[0] != None:
            r.append(tmp[0])
    return r

def countSection(spkB,secspkB):
    if len(secspkB) == 0 :
        return 0
    return secspkB.count(spkB) / len(secspkB)



    
        
        
    
    

i = pretraitement(ISMeeting())
AddTimeBySpeaker(i)
AddBestSpeaker(i)
t =  pretraitement(tSMeeting())
AddTimeBySpeaker(t)
AddBestSpeaker(t)

tot = []
tot.extend( ISMeeting())
tot.extend( tSMeeting())
c =  pretraitement(tot)
AddTimeBySpeaker(c)
AddBestSpeaker(c)
AddRole(c, 0.33, 5*60, 0.1, 0.5)
print(c)

ib = bestSpkRepartition(i,0.15)
print(ib)
tb = bestSpkRepartition(t,0.15)
print(tb)
totb = bestSpkRepartition(c,0.15)
print(totb)
    