#Extrait les features depuis les fichiers audio contenus dans le dossier précisé dans la variable path.

from numpy.ma import nomask
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import matplotlib.pyplot as plt

from os import listdir #Travail sur l'enregistrement des plots
from os.path import isfile, join
import shutil
import os

nomFeatures = ["Zero Crossing Rate","Energy","Entropy of Energy","Spectral Centroid","Spectral Spread","Spectral Entropy","Spectral Flux","Spectral Rolloff"]

path = "./lesCoupes"

#TODO : Voir pour eventuellement renvoyé une feature précise.
#TODO : csv

def extraireFeature(nomFichier,numero):

    nomDossierRes = "resultatCoupe" + str(numero) + "/"

    dir = "./" + "resultatCoupe" + str(numero)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    #Lecture du fichier audio -> fichier .wav a la base
    [Fs, x] = audioBasicIO.read_audio_file("./lesCoupes/"+fichier)

    #Extraction des shortTermFeatures dans un tableau, a voir si il faut pas utiliser le midTermFeatures plus tard.
    #frame size of 50 msecs and a frame step of 25 msecs
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)

    #Extraction de toutes les features  : https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction

    for i in range(7):

        print("feature",nomFeatures[i])
        # plt.subplot(2, 1, i + 1);
        plt.plot(F[i,:]);plt.xlabel('Frame no'); plt.ylabel(f_names[i])
        plt.title(nomFeatures[i])

        plt.savefig(nomDossierRes+nomFeatures[i]+'.png')

        plt.show()


        # plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

    for i in range(9,21):

        st = "MFCCs "+str(i)

        plt.plot(F[i,:]);plt.xlabel('Frame no'); plt.ylabel(f_names[i])
        plt.title(st)
        plt.savefig(nomDossierRes+st+".png")
        plt.show()

    for i in range(22,33):

        st = "Chroma Vector "+str(i)

        plt.plot(F[i,:]);plt.xlabel('Frame no'); plt.ylabel(f_names[i])
        plt.title(st)
        plt.savefig(nomDossierRes+st+".png")
        plt.show()

    plt.plot(F[34,:]);plt.xlabel('Frame no'); plt.ylabel(f_names[34])
    plt.title("Chroma deviation")
    plt.savefig(nomDossierRes+"Chroma_deviation.png")
    plt.show()


onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

cpt = 1

for fichier in onlyfiles:
    if ".wav" in fichier:
        print("Travail :",fichier)
        extraireFeature(fichier,cpt)
        cpt += 1