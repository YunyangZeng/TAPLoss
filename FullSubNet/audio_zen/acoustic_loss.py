import torch
import os
import torch.nn.functional as F


class AcousticLoss(torch.nn.Module):
    def __init__(self, args, device = 'cuda'):
        super(AcousticLoss, self).__init__()
        self.args = args
        self.device = device
        acoustic_model_path = self.args["acoustic_loss"]["model_path"] 
        model_state_dict = torch.load(acoustic_model_path, map_location=device)['model_state_dict']
        self.estimate_acoustics = AcousticEstimator()
        self.loss_type = self.args["acoustic_loss"]["type"]
        if self.loss_type == "l2":
            self.l2 = torch.nn.MSELoss()
        elif self.loss_type == "l1":
            self.l1 = torch.nn.L1Loss()
        self.estimate_acoustics.load_state_dict(model_state_dict)
        self.estimate_acoustics.to(device)
                
    def __call__(self, clean_waveform, enhan_wav_waveform, mode = "train"):
        return self.forward(clean_waveform, enhan_waveform, mode)

    def forward(self, clean_waveform, enhan_waveform, mode):    
        if mode == "train":
            self.estimate_acoustics.train()
        else:
            self.estimate_acoustics.eval()    
        clean_spectrogram = self.get_stft(clean_waveform)
        enhan_spectrogram,enhan_st_energy = self.get_stft(enhan_waveform, return_short_time_energy = True)        
        clean_acoustics = self.estimate_acoustics(clean_spectrogram)
        enhan_acoustics = self.estimate_acoustics(enhan_spectrogram)      
        loss_type = self.args["acoustic_loss"]["type"]
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
        return acoustic_loss
    
    def get_stft(self, wav, return_short_time_energy = False):
        self.nfft = 512
        self.hop_length = 160
        spec = torch.stft(wav, n_fft=self.nfft, hop_length=self.hop_length, return_complex=False)
        spec_real = spec[..., 0]
        spec_imag = spec[..., 1]     
        spec = spec.permute(0, 2, 1, 3).reshape(spec.size(dim=0), -1, 2 * (self.nfft//2 + 1))
        if return_short_time_energy:
            st_energy = torch.mul(torch.sum(spec_real**2 + spec_imag**2, dim = 1), 2/self.nfft)
            assert spec.size(dim=1) == st_energy.size(dim=1)
            return spec.float(), st_energy.float()
        else: 
            return spec.float()
                  
                                                 
class AcousticEstimator(torch.nn.Module):
    def __init__(self):
        super(AcousticEstimator, self).__init__()
        self.lstm = torch.nn.LSTM(514, 256, 3, bidirectional=True, batch_first=True)
        self.linear1 = torch.nn.Linear(512, 256)
        self.linear2 = torch.nn.Linear(256, 25)
        self.act = torch.nn.ReLU()
        
    def forward(self, A0):
        A1, _ = self.lstm(A0)
        Z1    = self.linear1(A1)
        A2    = self.act(Z1)
        Z2    = self.linear2(A2)
        return Z2

