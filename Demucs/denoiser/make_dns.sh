#python3 -m denoiser.audio /media/yunyangz/DNS/wav/30s/00-10/clean > egs/dns/tr/clean.json
#python3 -m denoiser.audio /media/yunyangz/DNS/wav/30s/00-10/noisy > egs/dns/tr/noisy.json

python3 -m denoiser.audio /media/konan/DNS/wav/30s/50-60/clean > egs/dns/cv/clean.json
python3 -m denoiser.audio /media/konan/DNS/wav/30s/50-60/noisy > egs/dns/cv/noisy.json

#python3 -m denoiser.audio /home/yunyangz/Documents/test_set/test_set/synthetic/no_reverb/clean > egs/dns/tt/clean.json
#python3 -m denoiser.audio /home/yunyangz/Documents/test_set/test_set/synthetic/no_reverb/noisy > egs/dns/tt/noisy.json

#python3 -m denoiser.audio /home/yunyangz/Documents/test_set/resynth_with_silence/clean > egs/dns/resynth_with_silence/clean.json
#python3 -m denoiser.audio /home/yunyangz/Documents/test_set/resynth_with_silence/noisy > egs/dns/resynth_with_silence/noisy.json

#python3 -m denoiser.audio /home/yunyangz/Documents/test_set/resynth_no_silence/clean > egs/dns/resynth_no_silence/clean.json
#python3 -m denoiser.audio /home/yunyangz/Documents/test_set/resynth_no_silence/noisy > egs/dns/resynth_no_silence/noisy.json

#python3 -m denoiser.audio /home/yunyangz/Documents/test_set/synthesized_data_0707/no_sil/WAV/train/clean > egs/dns/resynth_0707_no_silence/clean.json
#python3 -m denoiser.audio /home/yunyangz/Documents/test_set/synthesized_data_0707/no_sil/WAV/train/noisy > egs/dns/resynth_0707_no_silence/noisy.json

#python3 -m denoiser.audio /home/yunyangz/Documents/test_set/synthesized_data_0707/0.2_sil/WAV/train/clean > egs/dns/resynth_0707_0.2_silence/clean.json
#python3 -m denoiser.audio /home/yunyangz/Documents/test_set/synthesized_data_0707/0.2_sil/WAV/train/noisy > egs/dns/resynth_0707_0.2_silence/noisy.json