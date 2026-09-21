# Model output

`yamnet.tflite` is the default AudioSet-pretrained baseline used for immediate live monitoring. `python -m training.train` writes `battlefield_sound_cnn.pt` here; when present, that custom checkpoint takes priority over YAMNet.
