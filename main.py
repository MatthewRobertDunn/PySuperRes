import numpy as np
import tensorflow as tf
import scipy.io.wavfile
from tensorflow import keras
from tensorflow.keras import layers
import audio
import sounddevice as sd
import train
import glob


model = train.MyModel()
model.compile()
model.load()
#filename = r"C:\Users\Matthew\Downloads\WAProd_Free_Anniversary_Collection_Vol6\Demo Packs\What About Bass House District\Drum Loops\WABHD_125_Drum_Loop_30_Full.wav"
#model.train(filename)
def train():
  root_dir = "C:\\Users\\Matthew\\Downloads\\WAProd_Free_Anniversary_Collection_Vol6\\"
  print(root_dir)
  for filename in glob.iglob(root_dir + '**/*.wav', recursive=True):
       model.train(filename)

#train()

filename = r"c:\temp\omf.wav"
input = audio.get_input_data(filename)
out = model.upscale(input)

scipy.io.wavfile.write(r"c:\temp\omf.downscaled2.wav",16000,audio.undo_output(input))
scipy.io.wavfile.write(r"c:\temp\omf.upscaled2.wav",48000,out)
sd.play(out, samplerate=48000, blocking=True)






#print(model)
#sd.play(source,samplerate=inrate,blocking=True)
#sd.play(dest,samplerate=outrate,blocking=True)
