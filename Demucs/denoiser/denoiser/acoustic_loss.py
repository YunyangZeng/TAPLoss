import torch
import os
import torch.nn.functional as F
DEFAULT_MODEL_DIR  = "/home/yunyangz/Documents/Demucs/with_acoustic_loss/LLD_Estimator_STFT/ckpts/"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR + "lld-estimation-model_12mse_14mae.pt"


class AcousticLoss(torch.nn.Module):
    
    def __init__(self, args, acoustic_model_path = DEFAULT_MODEL_PATH, device = 'cuda'):
        
        super(AcousticLoss, self).__init__()
        model_state_dict = torch.load(acoustic_model_path, map_location=device)['model_state_dict']
        self.args = args
        self.estimate_acoustics = AcousticEstimator()
        if self.args is not None:
            if self.args.ac_loss_type == "matrix_l2":
                self.matrix_l2 = torch.nn.MSELoss()
            if self.args.ac_loss_type == "matrix_l1":
                self.matrix_l1 = torch.nn.L1Loss()
        self.estimate_acoustics.load_state_dict(model_state_dict)
        self.estimate_acoustics.to(device)
        self.estimate_acoustics.train()
        
    def __call__(self, clean_waveform, enhan_waveform):
        
        return self.forward(clean_waveform, enhan_waveform)

    def forward(self, clean_waveform, enhan_waveform, noisy_waveform = None):
        
        clean_spectrogram = self.get_stft(clean_waveform)
        enhan_spectrogram, enhan_st_energy = self.get_stft(enhan_waveform, return_short_time_energy = True)
        
        
        clean_acoustics = self.estimate_acoustics(clean_spectrogram)
        enhan_acoustics = self.estimate_acoustics(enhan_spectrogram)
        
        
        if noisy_waveform is not None:
            noisy_spectrogram = self.get_stft(noisy_waveform)
            noisy_acoustics  = self.estimate_acoustics(noisy_spectrogram)
            
            
        if self.args is None:
            """
                If you want to return the estimated acoustics, parse None to agrs
            """
            
            if noisy_waveform is not None:
                return {"clean_acoustics": clean_acoustics, "enhan_acoustics": enhan_acoustics, "noisy_acoustics": noisy_acoustics}
            else:
                return {"clean_acoustics": clean_acoustics, "enhan_acoustics": enhan_acoustics}
            
            
        else:

            if self.args.ac_loss_type == "l2":
                acoustic_loss   = self.matrix_l2(enhan_acoustics, clean_acoustics)
            elif self.args.ac_loss_type == "l1":
                acoustic_loss   = self.matrix_l1(enhan_acoustics, clean_acoustics)
            elif self.args.ac_loss_type == "frame_energy_weighted_l2":
                acoustic_loss   = torch.mean(((torch.sigmoid(enhan_st_energy)** 0.5).unsqueeze(dim = -1) \
                * (enhan_acoustics - clean_acoustics)) ** 2 )                                       
            elif self.args.ac_loss_type == "frame_energy_weighted_l1":
                acoustic_loss   = torch.mean(torch.sigmoid(enhan_st_energy).unsqueeze(dim = -1) \
                * torch.abs(enhan_acoustics - clean_acoustics))
                
            return acoustic_loss            
           
    
    def get_stft(self, wav, return_short_time_energy = False):
        self.nfft = 512
        self.hop_length = 160
        spec = torch.stft(wav, n_fft=self.nfft, hop_length=self.hop_length, return_complex=False)
        
        
        spec_real = spec[..., 0]
        spec_imag = spec[..., 1]
             
                
        spec = spec.permute(0, 2, 1, 3).reshape(spec.size(dim=0), -1, 514)

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

