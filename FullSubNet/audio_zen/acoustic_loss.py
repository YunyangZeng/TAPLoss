import torch
import os
import torch.nn.functional as F


class AcousticLoss(torch.nn.Module):
    
    def __init__(self, return_LLDs = None, acoustic_model_path = None, device = 'cuda'):
        
        super(AcousticLoss, self).__init__()
        model_state_dict = torch.load(acoustic_model_path, map_location=device)['model_state_dict']
        self.return_LLDs = return_LLDs
        self.estimate_acoustics = AcousticEstimator()
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.estimate_acoustics.load_state_dict(model_state_dict)
        self.estimate_acoustics.to(device)
        #self.estimate_acoustics.train()
            
        
    def __call__(self, clean_spectrogram, enhan_spectrogram, noisy_spectrogram = None, mode = "train",loss_type = "l1"):
        
        return self.forward(clean_spectrogram, enhan_spectrogram, noisy_spectrogram, mode, loss_type)

    def forward(self, clean_spectrogram, enhan_spectrogram, noisy_spectrogram, mode, loss_type):
        
        
        """ 
            clean_stft => (B, T, F * 2) 
            enhan_stft => (B, T, F * 2)
            noisy_stft => (B, T, F * 2)
            real and imag spec are conatenated at the last dimension
        
        """
        if mode == "train":
            self.estimate_acoustics.train()
        else:
            self.estimate_acoustics.eval()
        
        enhan_st_energy = self.get_st_energy(enhan_spectrogram)
        clean_acoustics = self.estimate_acoustics(clean_spectrogram)
        enhan_acoustics = self.estimate_acoustics(enhan_spectrogram)
        
        if noisy_spectrogram is not None:
            noisy_acoustics = self.estimate_acoustics(noisy_spectrogram)
            
            
        if self.return_LLDs:
            if noisy_spectrogram is not None:
                return {"clean_acoustics": clean_acoustics, "enhan_acoustics": enhan_acoustics, "noisy_acoustics": noisy_acoustics}
            else:
                return {"clean_acoustics": clean_acoustics, "enhan_acoustics": enhan_acoustics}
        else:
            """
            L1
            """
            #acoustic_loss   = self.l1_loss(enhan_acoustics, clean_acoustics)

            if loss_type == "l1":
                acoustic_loss   = torch.mean(torch.sigmoid(enhan_st_energy).unsqueeze(dim = -1) \
                * torch.abs(enhan_acoustics - clean_acoustics))

            elif loss_type == "l2":
                acoustic_loss   = torch.mean(((torch.sigmoid(enhan_st_energy)** 0.5).unsqueeze(dim = -1) \
                * (enhan_acoustics - clean_acoustics)) ** 2 )

            return acoustic_loss
    
    def get_stft(self, wav):
        
        self.nfft = 512
        self.hop_length = self.nfft // 2
        
        spec = torch.stft(wav, n_fft=self.nfft, hop_length=self.hop_length, return_complex=False)

        spec = spec.permute(0, 2, 1, 3).reshape(spec.size(dim=0), -1, (self.nfft // 2 + 1) * 2)
        """
            spec_real ==> (B, T, 514)
        """
            
        return spec.float()
    def get_st_energy(self, spec):
        
                                              
        spec_real = spec[..., :spec.size(dim=-1)//2]
        spec_imag = spec[..., spec.size(dim=-1)//2:]                                         
        """
            spec_real ==> (B, T, 257)
            spec_imag ==> (B, T, 257)
        """
        st_energy = torch.mul(torch.sum(spec_real**2 + spec_imag**2, dim = 2), 2/spec.size(dim=-1))
         
        assert spec.size(dim=1) == st_energy.size(dim=1)
        return st_energy.float()                                     
                                                 
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

