import os
import torch
import torchaudio
from torch.utils.data import Dataset
from models.GE2E import CONFIG as GE2E_config

GE2E_config= GE2E_config

class FeatureExtractorDataset(Dataset):
    
    def __init__(self, model_name, data_dir, split):
        self.data_dir = os.path.join(data_dir,split)
        self.model = model_name
        self._index_audios()
        self.sampling_rate = 16000
        _, org_sample_rate = torchaudio.load(self.audio_files[0])
        self.resample_transform = torchaudio.transforms.Resample(org_sample_rate, self.sampling_rate)

        if self.model == 'GE2E':
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate = self.sampling_rate,
                n_fft = int(self.sampling_rate * GE2E_config.mel_window_length / 1000),
                hop_length = int(self.sampling_rate * GE2E_config.mel_window_step / 1000),
                n_mels = GE2E_config.mel_n_channels)

    def _index_audios(self):
        # Gather all audio file locations
        self.audio_files = []
        for audiofile in os.listdir(self.data_dir):
            filename = os.path.join(self.data_dir,audiofile)
            self.audio_files.append(filename)
        
    def __getitem__(self, idx):
        file_name = self.audio_files[idx]
        audio, sr = torchaudio.load(file_name)
        if sr != self.sampling_rate: audio = self.resample_transform(audio)
        if audio.size(0)>1: audio = audio.mean(dim=0)
        audio = audio.squeeze()
        if self.model == 'GE2E':
            audio = self.mel_transform(audio).T
        length = audio.size(0)
        return audio, length, file_name
    
    def __len__(self):
        return len(self.audio_files)

    def data_collator(self,batch):
        seq_lengths, file_paths = list(zip(*batch))[1:]
        max_seq_len = max(seq_lengths)

        if self.model == 'GE2E':
            num_channels = batch[0][0].size(1)
            collated_batch = torch.zeros((len(batch), max_seq_len, num_channels))
        
            for idx, sample in enumerate(batch):
                collated_batch[idx] = torch.cat([sample[0], torch.zeros((max_seq_len - seq_lengths[idx], num_channels))])
        else:
            collated_batch = torch.zeros((len(batch), max_seq_len))
            for idx, sample in enumerate(batch):
                collated_batch[idx] = torch.cat([sample[0], torch.zeros((max_seq_len - seq_lengths[idx]))])

        return collated_batch, file_paths



class MemoryModeClassifierDataset(Dataset):

    def __init__(self, fx_model_name, fx_model, device, data_dir, datainfo, split):
        self.data_dir = os.path.join(data_dir,split)
        self.device = 'cpu'
        self.model_name = fx_model_name
        self.model = fx_model
        self.model.to(self.device)
        
        self._index_audios()
        self.sampling_rate = 16000
        _, org_sample_rate = torchaudio.load(self.audio_files[0])
        self.resample_transform = torchaudio.transforms.Resample(org_sample_rate, self.sampling_rate)

        if self.model_name == 'GE2E':
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate = self.sampling_rate,
                n_fft = int(self.sampling_rate * GE2E_config.mel_window_length / 1000),
                hop_length = int(self.sampling_rate * GE2E_config.mel_window_step / 1000),
                n_mels = GE2E_config.mel_n_channels)

        self.label_map = datainfo.label_map
        self.label_indices = datainfo.label_indices
        self.label_to_id = {k:i for k,i in zip(self.label_map.keys(),[*range(len(self.label_map))])}
        self.id_to_label = {v:k for k,v in self.label_to_id.items()}

    def _index_audios(self):
        # Gather all audio file locations
        self.audio_files = []
        for audiofile in os.listdir(self.data_dir):
            filename = os.path.join(self.data_dir,audiofile)
            self.audio_files.append(filename)
        
    def __getitem__(self, idx):

        file_name = self.audio_files[idx]
        audio, sr = torchaudio.load(file_name)
        if sr != self.sampling_rate: audio = self.resample_transform(audio)
        if audio.size(0)>1: audio = audio.mean(dim=0)
        audio = audio.squeeze()
        if self.model_name == 'GE2E':
            audio = self.mel_transform(audio).T

        audio = audio.to(self.device)

        with torch.inference_mode():
            if self.model_name == 'GE2E':
                features = self.model(audio.unsqueeze(dim=0))
                features = features.squeeze()
            else:
                features, _ = self.model.extract_features(audio.unsqueeze(dim=0))
                features = torch.stack([*features],dim=1)
                features = features.squeeze()

        label = self.label_to_id[self.audio_files[idx].split('/')[-1][self.label_indices['from']:self.label_indices['to']]]
        if self.model_name == 'GE2E':  seq_length = len(features)
        else:  seq_length = features.shape[1]

        return features, seq_length, label

    def __len__(self):
        return len(self.audio_files)
    
    def data_collator(self,batch):
        features, seq_lengths, labels = zip(*batch)
        batch_size = len(batch)
        max_seq_len = max(seq_lengths)

        if self.model_name == 'GE2E':
            padded_features = torch.zeros(batch_size, max_seq_len)
            for idx, sample in enumerate(batch):
                padded_features[idx] = torch.cat([sample[0], torch.zeros((max_seq_len - seq_lengths[idx]))])
        else:
            layers = features[0].shape[0]
            feature_dim = features[0].shape[2]
            padded_features = torch.zeros((batch_size, layers, max_seq_len, feature_dim))
            for idx, sample in enumerate(batch):
                padded_features[idx,:,:seq_lengths[idx], :] = features[idx]

        labels = torch.tensor(labels)
        seq_lengths = torch.tensor(seq_lengths)

        return padded_features, seq_lengths, labels



class DiskModeClassifierDataset(Dataset):

    def __init__(self, fx_model_name, data_dir, datainfo, split):
        self.data_dir = os.path.join(data_dir,split)
        self.fx_model = fx_model_name
        self._index_features()

        self.label_map = datainfo.label_map
        self.label_indices = datainfo.label_indices
        self.label_to_id = {k:i for k,i in zip(self.label_map.keys(),[*range(len(self.label_map))])}
        self.id_to_label = {v:k for k,v in self.label_to_id.items()}

    def _index_features(self):
        # Gather all feature file locations
        self.feature_files = []
        for featurefile in os.listdir(self.data_dir):
            filename = os.path.join(self.data_dir,featurefile)
            self.feature_files.append(filename)
        
    def __getitem__(self, idx):
        feature = torch.load(self.feature_files[idx],map_location='cpu')
        label = self.label_to_id[self.feature_files[idx].split('/')[-1][self.label_indices['from']:self.label_indices['to']]]
        if self.fx_model == 'GE2E': seq_length = len(feature)
        else: seq_length = feature.shape[1]

        return feature, seq_length, label

    def __len__(self):
        return len(self.feature_files)
    
    def data_collator(self,batch):
        features, seq_lengths, labels = zip(*batch)
        batch_size = len(batch)
        max_seq_len = max(seq_lengths)

        if self.fx_model == 'GE2E':
            padded_features = torch.zeros(batch_size, max_seq_len)
            for idx, sample in enumerate(batch):
                padded_features[idx] = torch.cat([sample[0], torch.zeros((max_seq_len - seq_lengths[idx]))])
        else:
            layers = features[0].shape[0]
            feature_dim = features[0].shape[2]
            padded_features = torch.zeros((batch_size, layers, max_seq_len, feature_dim))
            for idx, sample in enumerate(batch):
                padded_features[idx,:,:seq_lengths[idx], :] = features[idx]

        labels = torch.tensor(labels)
        seq_lengths = torch.tensor(seq_lengths)

        return padded_features, seq_lengths, labels