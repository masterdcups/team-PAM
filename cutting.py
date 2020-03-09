#Decoupe un fichier audio en sous fichier.

from pydub import AudioSegment

file_path = "./IS1000a.Mix-Headset.wav"

audio = AudioSegment.from_file(file_path)
#TODO : Faire selon particiapnt,temps debut et fin et path.
longeurBout = 300000
ancien = 0
compteur = 1

print(audio.duration_seconds)

for i in range(5):

    newAudio = AudioSegment.from_wav(file_path)
    newAudio = newAudio[ancien:ancien+longeurBout]
    ancien += longeurBout

    newAudio.export("./lesCoupes/coupe"+str(compteur)+".wav", format="wav")
    compteur += 1

newAudio = AudioSegment.from_wav(file_path)
newAudio = newAudio[ancien:]

newAudio.export("./lesCoupes/coupe"+str(compteur)+".wav", format="wav")
