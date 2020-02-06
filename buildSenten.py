# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 08:38:44 2020

@author: Lenovo T420s
"""

from lxml import etree
import spacy

import localisationAndParam as locp


segmentLoc = locp.__groundT__ / "segments"
wordsLoc = locp.__groundT__ / "words"
nlp = spacy.load("en_core_web_lg") 
""" Extraction"""

def onePortionExtractor(part, speaker ):
    """
    segmentName = Meeting + part  + "." + speaker + ".segments.xml"
    print(segmentName)
    segmentTree = etree.parse(str(segmentLoc / segmentName))
    """
    cpt = 0
    sentenses = {}
    wordsName = locp.__Meeting + part  + "." + speaker + ".words.xml"
    wordsTree = etree.parse(str(wordsLoc / wordsName))
    begin = True
    s = ""
    for w in wordsTree.xpath("w"):
        if begin:
            begin = False
            b = w.get("starttime")
        s +=  w.text + " "
        if w.get("punc") == "true" and (w.text in [".","!","?"]) :
            sentenses[cpt] = [float(b),float(w.get("endtime")),s]
            s = ""
            begin = True
            cpt += 1
    return (sentenses)

def onePortionExtractorBis(part, speaker ):
    """
    segmentName = Meeting + part  + "." + speaker + ".segments.xml"
    print(segmentName)
    segmentTree = etree.parse(str(segmentLoc / segmentName))
    """
    cpt = 0
    sentenses = {}
    wordsName = locp.__Meeting + part  + "." + speaker + ".words.xml"
    wordsTree = etree.parse(str(wordsLoc / wordsName))
    s = ""
    for w in wordsTree.xpath("w"):
        b = w.get("starttime")
        s =  w.text
        cpt += 1
        sentenses[cpt] = [float(b),float(w.get("endtime")),s]
        cpt += 1
    return (sentenses)

def partExtractor(partName,speakers):
    part = {}
    for spker in speakers:        
        part[spker] = onePortionExtractorBis(partName,spker)
    return part

def sentensesAllSort(part):
    def valueOfElem(e):
        return(e[1]) 
    
    totalSp = []
    for  speaker, speach in part.items():
        for sent in speach.values():
            totalSp.append( [speaker] + sent)
    res = sorted(totalSp,key = valueOfElem)        
    return res
            
""" Part of speach detection"""

def groupSpeach(sortSents):
    res = []
    spker = sortSents[0][0]
    save = sortSents[0][3]
    b = sortSents[0][1]
    f = sortSents[0][2]
    for s in sortSents:
        if s[0] != spker:
            res.append((spker,b,f,save))
            spker = s[0]
            save = s[3]
            b = s[1]
            f = s[2]
        else:
            if not(s[3] in [",",".","!","?"]):
                save += " "
            save += s[3]
            f = s[2]

    res.append((spker,b,s[1],save))
    return res

def detectMonologue(grSents,minTime):

    monologues = []
    for s in grSents:
        if s[2]- s[1] > minTime:
            monologues.append(s)
    return monologues
            
"""   
def interuption(ordSP, minTime):
    return 1
    
def dialogue(ordSP,minTime):
    monologues = []
    spker = [sortSents[0][0]]
    save = [sortSents[0][3]] 
    b = { }
    for s in sortSents:
        if s[0] != spker:
            if (s[1] - b) > minTime:
                monologues.append((spker,b,s[1],save))
            spker = s[0]
            save = [s[3]]
            b = s[1]
        else:
            save.append(s[3])
    if  sortSents[-1][1] - b > minTime:
        monologues.append((spker,b,s[1],save))
    return monologues
"""
"""sementics"""
nlp = spacy.load("en_core_web_lg") 
def querryIdent(grSents):
    qw = [[],[]]
    for sent in grSents:
        docSent = nlp(sent[3])
        for sent in docSent.sents:
            doc = nlp(sent.text)
            q = False
            Subjfirst =  False
            VerbFirst = False
            sub = False
            verb = False
            for token in doc:
                q = token.text == "?" or q
                if token.dep_ == "nsubj" and not sub :
                    Subjfirst = not VerbFirst
                    sub = True
                elif token.dep_ == "ROOT" and not verb:
                    VerbFirst = not Subjfirst
                    verb = True
                
            if (VerbFirst  and (sub and verb)):
                qw[0].append(sent.text)
            elif  (q and sub and verb) :
                qw[1].append(sent.text)
                
            
            
    return qw

def speachToSent(sp):
    sents= []

    s = ""
    doc = nlp(sp)
    for token in doc:
        s += token.text + " "
        if token.text in [".","!","?"]:
            sents.append(s)
            s = ""
    if s != "":
        sents.append(s)
    return sents
         
def organisedSpeachTotalPipe(partOfSpeach = "a"):
    return groupSpeach(sentensesAllSort(partExtractor(partOfSpeach,locp.__seapkers)))
        
            
            

p = partExtractor("a",locp.__seapkers)
o = sentensesAllSort(p)
m = groupSpeach(o)
"""
print(detectMonologue(m,100))
res = querryIdent(m)
print(res[0])
print(res[1])
"""


