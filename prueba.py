import numpy as np
import pyaudio as pa
import os, time
import librosa
#import IPython.display as ipd
import matplotlib.pyplot as plt
#%matplotlib inline

import nemo
import nemo.collections.asr as nemo_asr


# sample rate, Hz
SAMPLE_RATE = 16000


mbn_model = nemo_asr.models.EncDecClassificationModel.from_pretrained("commandrecognition_en_matchboxnet3x1x64_v2")	
vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained('vad_marblenet')

from omegaconf import OmegaConf
import copy



# Preserve a copy of the full config
vad_cfg = copy.deepcopy(vad_model._cfg)
mbn_cfg = copy.deepcopy(mbn_model._cfg)
print(OmegaConf.to_yaml(mbn_cfg))


labels = mbn_cfg.labels
for i in range(len(labels)):
    print('%-10s' % (labels[i]), end=' ')


# Set model to inference mode
mbn_model.eval();
vad_model.eval();


from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch
from torch.utils.data import DataLoader

# simple data layer to pass audio signal
class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)
        
    def set_signal(self, signal):
        self.signal = signal.astype(np.float32)/32768.
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1


data_layer = AudioDataLayer(sample_rate=mbn_cfg.train_ds.sample_rate)
data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)


def infer_signal(model, signal):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    audio_signal, audio_signal_len = audio_signal.to(model.device), audio_signal_len.to(model.device)
    logits = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
    return logits

# class for streaming frame-based ASR
# 1) use reset() method to reset FrameASR's state
# 2) call transcribe(frame) to do ASR on
#    contiguous signal's frames
class FrameASR:
    
    def __init__(self, model_definition,
                 frame_len=2, frame_overlap=2.5, 
                 offset=0):
        '''
        Args:
          frame_len (seconds): Frame's duration
          frame_overlap (seconds): Duration of overlaps before and after current frame.
          offset: Number of symbols to drop for smooth streaming.
        '''
        self.task = model_definition['task']
        self.vocab = list(model_definition['labels'])
        
        self.sr = model_definition['sample_rate']
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        timestep_duration = model_definition['AudioToMFCCPreprocessor']['window_stride']
        for block in model_definition['JasperEncoder']['jasper']:
            timestep_duration *= block['stride'][0] ** block['repeat']
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset
        self.reset()
        
    @torch.no_grad()
    def _decode(self, frame, offset=0):
        assert len(frame)==self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame

        if self.task == 'mbn':
            logits = infer_signal(mbn_model, self.buffer).to('cpu').numpy()[0]
            decoded = self._mbn_greedy_decoder(logits, self.vocab)
            
        elif self.task == 'vad':
            logits = infer_signal(vad_model, self.buffer).to('cpu').numpy()[0]
            decoded = self._vad_greedy_decoder(logits, self.vocab)
           
        else:
            raise("Task should either be of mbn or vad!")
            
        return decoded[:len(decoded)-offset]
    
    def transcribe(self, frame=None,merge=False):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        unmerged = self._decode(frame, self.offset)
        return unmerged
        
    
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.mbn_s = []
        self.vad_s = []
        
    @staticmethod
    def _mbn_greedy_decoder(logits, vocab):
        mbn_s = []
        if logits.shape[0]:
            class_idx = np.argmax(logits)
            class_label = vocab[class_idx]
            mbn_s.append(class_label)         
        return mbn_s
    
    
    @staticmethod
    def _vad_greedy_decoder(logits, vocab):
        vad_s = []
        if logits.shape[0]:
            probs = torch.softmax(torch.as_tensor(logits), dim=-1)
            probas, preds = torch.max(probs, dim=-1)
            vad_s = [preds.item(), str(vocab[preds]), probs[0].item(), probs[1].item(), str(logits)]
        return vad_s

vad_threshold = 0.8 

STEP = 0.1 
WINDOW_SIZE = 0.15
mbn_WINDOW_SIZE = 1

CHANNELS = 1 
RATE = SAMPLE_RATE
FRAME_LEN = STEP # use step of vad inference as frame len

CHUNK_SIZE = int(STEP * RATE)
vad = FrameASR(model_definition = {
                   'task': 'vad',
                   'sample_rate': SAMPLE_RATE,
                   'AudioToMFCCPreprocessor': vad_cfg.preprocessor,
                   'JasperEncoder': vad_cfg.encoder,
                   'labels': vad_cfg.labels
               },
               frame_len=FRAME_LEN, frame_overlap=(WINDOW_SIZE - FRAME_LEN) / 2, 
               offset=0)

mbn = FrameASR(model_definition = {
                       'task': 'mbn',
                       'sample_rate': SAMPLE_RATE,
                       'AudioToMFCCPreprocessor': mbn_cfg.preprocessor,
                       'JasperEncoder': mbn_cfg.encoder,
                       'labels': mbn_cfg.labels
                   },
                   frame_len=FRAME_LEN, frame_overlap = (mbn_WINDOW_SIZE-FRAME_LEN)/2,
                   offset=0)

vad.reset()
mbn.reset()

# Setup input device
p = pa.PyAudio()
print('Available audio input devices:')
input_devices = []
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('maxInputChannels'):
        input_devices.append(i)
        print(i, dev.get('name'))

if len(input_devices):
    dev_idx = -2
    while dev_idx not in input_devices:
        print('Please type input device ID:')
        dev_idx = int(input())

    
    def callback(in_data, frame_count, time_info, status):
        """
        callback function for streaming audio and performing inference
        """
        signal = np.frombuffer(in_data, dtype=np.int16)
        vad_result = vad.transcribe(signal) 
        mbn_result = mbn.transcribe(signal) 
        
        if len(vad_result):
            # if speech prob is higher than threshold, we decide it contains speech utterance 
            # and activate MatchBoxNet 
            if vad_result[3] >= vad_threshold: 
                print(mbn_result) # print mbn result when speech present
            else:
                print("no-speech")
        return (in_data, pa.paContinue)

    # streaming
    stream = p.open(format=pa.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=dev_idx,
                    stream_callback=callback,
                    frames_per_buffer=CHUNK_SIZE)

    
    print('Listening...')
    stream.start_stream()
    
    # Interrupt kernel and then speak for a few more words to exit the pyaudio loop !
    try:
        while stream.is_active():
            time.sleep(0.1)
    finally:        
        stream.stop_stream()
        stream.close()
        p.terminate()
        print()
        print("PyAudio stopped")
    
else:
    print('ERROR: No audio input device found.')
