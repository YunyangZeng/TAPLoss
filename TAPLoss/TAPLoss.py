import torch
import os
import sys
import torch.nn.functional as F
import numpy as np
from TAP_estimator import AcousticEstimator

class AcousticLoss(torch.nn.Module):
    def __init__(self, loss_type, acoustic_model_path, paap = False, paap_weight_path = None, device = 'cuda'):
        super(AcousticLoss, self).__init__()
        self.device      = device
        self.paap        = paap
        model_state_dict = torch.load(acoustic_model_path, map_location=device)['model_state_dict']
        self.estimate_acoustics = AcousticEstimator()
        self.loss_type = loss_type
        if self.loss_type == "l2":
            self.l2 = torch.nn.MSELoss()
        elif self.loss_type == "l1":
            self.l1 = torch.nn.L1Loss()    
        if paap:
            if paap_weight_path is None:
                raise ValueError("PAAP weight path is not provided")
            else:
                self.paap_weight = torch.from_numpy(np.load(paap_weight_path)).to(device)
        self.estimate_acoustics.load_state_dict(model_state_dict)
        self.estimate_acoustics.to(device)
        
    def __call__(self, clean_waveform, enhan_waveform, mode="train"):
        return self.forward(clean_waveform, enhan_waveform, mode)

    def forward(self, clean_waveform, enhan_waveform, mode):
        if mode == "train":
            self.estimate_acoustics.train()
        else:
            self.estimate_acoustics.eval()        
        clean_spectrogram = self.get_stft(clean_waveform)
        enhan_spectrogram, enhan_st_energy = self.get_stft(enhan_waveform, return_short_time_energy = True)
        clean_acoustics   = self.estimate_acoustics(clean_spectrogram)
        enhan_acoustics   = self.estimate_acoustics(enhan_spectrogram)
        if self.paap:
            """
                paap_weight ==> (26, 40) 
                acoustics ==> (B, T, 25), expand last dimension by 1 for bias
            """
            clean_acoustics = torch.cat((clean_acoustics, torch.ones(clean_acoustics.size(dim = 0),\
                        clean_acoustics.size(dim = 1), 1, device = self.device)), dim = -1) # acoustics ==> (B, T, 26)
            
            enhan_acoustics = torch.cat((enhan_acoustics, torch.ones(enhan_acoustics.size(dim = 0),\
                        enhan_acoustics.size(dim = 1), 1, device = self.device)), dim = -1) # acoustics ==> (B, T, 26)
            
            clean_acoustics = clean_acoustics @ self.paap_weight # acoustics ==> (B, T, 40)
            enhan_acoustics = enhan_acoustics @ self.paap_weight # acoustics ==> (B, T, 40)  
        """
        loss_type must be one of the following 4 options:
        ["l2", "l1", "frame_energy_weighted_l2", "frame_energy_weighted_l1"]
        """
        if self.loss_type == "l2":
            acoustic_loss   = self.l2(enhan_acoustics, clean_acoustics)
        elif self.loss_type == "l1":
            acoustic_loss   = self.l1(enhan_acoustics, clean_acoustics)
        elif self.loss_type == "frame_energy_weighted_l2":
            acoustic_loss   = torch.mean(((torch.sigmoid(enhan_st_energy)** 0.5).unsqueeze(dim = -1) \
            * (enhan_acoustics - clean_acoustics)) ** 2 )                                       
        elif self.loss_type == "frame_energy_weighted_l1":
            acoustic_loss   = torch.mean(torch.sigmoid(enhan_st_energy).unsqueeze(dim = -1) \
            * torch.abs(enhan_acoustics - clean_acoustics))
        else:
            raise ValueError(f"Invalid loss_type {self.loss_type}")
        return acoustic_loss            
           
    
    def get_stft(self, wav, return_short_time_energy = False):
        self.nfft       = 512
        self.hop_length = 160
        spec = torch.stft(wav, n_fft=self.nfft, hop_length=self.hop_length, return_complex=False)
        spec_real = spec[..., 0]
        spec_imag = spec[..., 1]     
        spec = spec.permute(0, 2, 1, 3).reshape(spec.size(dim=0), -1,  2 * (self.nfft//2 + 1))

        if return_short_time_energy:
            st_energy = torch.mul(torch.sum(spec_real**2 + spec_imag**2, dim = 1), 2/self.nfft)
            assert spec.size(dim=1) == st_energy.size(dim=1)
            return spec.float(), st_energy.float()
        else: 
            return spec.float()
        


