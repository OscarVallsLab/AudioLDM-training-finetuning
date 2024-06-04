import torch

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import soundfile as sf

from transformers import EncodecModel, AutoProcessor


class Encodec(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encodec = EncodecModel.from_pretrained("facebook/encodec_32khz")
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")


    def encode(self,batch):
        ## Prepare data structure --> Batched audios are passed as a list of ndarrays
        unbinded_batch = torch.unbind(batch.squeeze(dim=1).cpu(),dim=0)
        list_batch = [sample.cpu().numpy() for sample in unbinded_batch]

        ## Encoder forward pass
        inputs = self.processor(raw_audio=list_batch, sampling_rate=self.processor.sampling_rate, return_tensors="pt").to(self.device)
        encoder_outputs = self.encodec.encode(inputs["input_values"], inputs["padding_mask"])

        ## Keep relevant information
        self.audio_scales = encoder_outputs.audio_scales
        self.padding_mask = inputs['padding_mask']

        return encoder_outputs.audio_codes.squeeze(0).unsqueeze(-1)
    
    def decode(self,batch):
        batch = batch.squeeze(-1).unsqueeze(0)
        audio_values = self.encodec.decode(audio_codes=batch,audio_scales=self.audio_scales,padding_mask=self.padding_mask)
        return audio_values
