# TODO: Make repo
# TODO: Make gitignore
# TODO: Add whisper
# TODO: Add BYOL-a /BYOL-s
# TODO: 


import os
import json
import argparse
import datetime
from tqdm import tqdm
from time import sleep
from models.GE2E import GE2E
from models.Dense import Dense

from utils.data import DotDict
from utils.logger import LEVELS as log_levels
from utils.logger import NoFmtLog, get_logger
from torcheval.metrics import MulticlassAccuracy
from tqdm.contrib.logging import logging_redirect_tqdm
from datasets import FeatureExtractorDataset, DiskModeClassifierDataset, MemoryModeClassifierDataset


import torch
import torchaudio
from torch.utils.data import DataLoader

VERSION = '1.1'

FX_MODELS = ['WAV2VEC2_BASE','WAV2VEC2_LARGE',
        'WAV2VEC2_BASE_XLSR','WAV2VEC2_LARGE_XLSR',
        'HUBERT_BASE', 'HUBERT_LARGE',
        'GE2E']

FX_MODEL_MAP = {'WAV2VEC2_BASE':'WAV2VEC2_BASE','WAV2VEC2_LARGE':'WAV2VEC2_LARGE',
        'WAV2VEC2_BASE_XLSR':'WAV2VEC2_XLSR53','WAV2VEC2_LARGE_XLSR':'WAV2VEC2_XLSR_300M',
        'HUBERT_BASE':'HUBERT_BASE', 'HUBERT_LARGE':'HUBERT_LARGE'}

DATASETS = ['AESDD','CaFE','EmoDB','EMOVO','IEMOCAP','RAVDESS','ShEMO']


class Trainer:

    def __init__(self, config):

        self.config = config

        # Setup Directories
        self.data_dir = os.path.join(config.data_dir, 'Audios', config.dataset)
        self.feature_dir = os.path.join(config.data_dir, 'Features', config.dataset, self.config.model)
        self.history_dir = os.path.join(config.history_dir, f'v{VERSION}',config.dataset, config.model, config.run_name)
        self.weights_dir = os.path.join(config.weights_dir, f'v{VERSION}',config.dataset, config.model, config.run_name)
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.history_dir)
        os.makedirs(self.weights_dir)

        # Setup logger
        log_file = os.path.join(self.history_dir, f'std.log')
        self.logger, self.no_fmt_logger = get_logger(config.run_name, config.log_level, log_file)
        self.no_fmt_log = NoFmtLog(self.no_fmt_logger)

        # Log configs
        self._print_banner()

        # Config check
        self._config_check()
    
        # Setup device
        if self.config.device == 'gpu': 
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else: self.device = 'cpu'
        self.logger.info(f'Running on {str(self.device).upper()}')
        self.no_fmt_log()

        # Load data info
        self.dataset_info = DotDict(json.load(open(os.path.join(self.data_dir,'info.json'))))
        
        # Setup models
        self.fx_model = self._get_feature_extractor()
        self.clf_model = Dense(self.config.model, len(self.dataset_info.label_map), self.device)
        
        # Setup metric
        self.acc_metric = MulticlassAccuracy(device=self.device)
        
        # Setup history
        self.history = DotDict(train_loss=[],train_acc=[],val_loss=[],val_acc=[],test_loss=[],test_acc=[])
        

    def _print_banner(self):
        try: console_width, _ = os.get_terminal_size(0)
        except : console_width = 50

        banner = '\n'+('-'*console_width)+f'\nMultilingual SER System v{VERSION}\n'+ ('-'*console_width) + '\n'
        max_len = max([len(key) for key in self.config.__dict__.keys()])+2
        for arg,val in self.config.__dict__.items():
            banner += ' '.join(arg.split('_')).title() + ' '*(max_len-len(arg))+': '+ str(val) + '\n'

        banner += f'\nTimestamp : {datetime.datetime.now()}\n'
        self.no_fmt_log(msg=banner)
        

    def _config_check(self):
        # Check for model
        if self.config.model not in FX_MODELS:
            self.logger.critical(f': No implementation found for {self.config.model} feature extractor\n\
                Available models: {FX_MODELS}')
            raise NotImplementedError(f': No implementation found for {self.config.model} feature extractor\n\
                Available models: {FX_MODELS}')
        # Check for dataset
        if self.config.dataset not in DATASETS:
            self.logger.critical(f': No implementation found for {self.config.dataset} dataset\n\
                Available datasets: {DATASETS}')
            raise NotImplementedError(f': No implementation found for {self.config.dataset} dataset\n\
                Available datasets: {DATASETS}')


    def _get_fx_dataloaders(self):
        test_dataset = FeatureExtractorDataset(self.config.model, self.data_dir, 'test')
        train_dataset = FeatureExtractorDataset(self.config.model, self.data_dir, 'train')
        val_dataset = FeatureExtractorDataset(self.config.model, self.data_dir, 'validation')
        
        train_batch_size = 32
        test_batch_size = 32
        num_workers = 2

        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=num_workers, collate_fn = test_dataset.data_collator)
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size, shuffle=False,
            num_workers=num_workers, collate_fn = train_dataset.data_collator)
        val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=num_workers, collate_fn = val_dataset.data_collator)

        return DotDict(train=train_loader, test=test_loader, validation=val_loader)


    def _get_feature_extractor(self):
        # Load Model
        try:
            if self.config.model == 'GE2E':
                model = GE2E()
                checkpoint = torch.load(os.path.join(self.config.weights_dir,'pretrained',self.config.model+'_weights.pt'), 'cpu')
                model.load_state_dict(checkpoint["model_state"])
            else:
                bundle = getattr(torchaudio.pipelines, FX_MODEL_MAP[self.config.model])
                model = bundle.get_model().to(self.device)
        except:
            self.logger.exception(f"Could not load {self.config.model} feature extractor")
            exit(0)
        return model


    def _extract_features(self):
        self.fx_model.to(self.device)
        self.fx_model.eval()

        processing_required = False
        processing_lock = os.path.join(self.feature_dir,'processing.lock')
          
        # Check if extraction is running
        while os.path.isfile(processing_lock): 
            self.logger.info('Waiting for other process to finish processing')
            sleep(60)
        else: self.no_fmt_log()
        
        # Check if extraction is required
        for split, dataloader in self.fx_dataloaders.items(): 
            if os.path.isdir(os.path.join(self.feature_dir,split)):
                self.logger.info(f'Feature cache found for {split} split')
            else: 
                self.logger.info(f'Feature cache not found for {split} split')
                processing_required = True
                
        if self.config.purge_cache and processing_required==False:
            os.system(f'rm -rf {self.feature_dir}')
            os.makedirs(self.feature_dir)
            self.logger.info('Feature cache purged')
            processing_required = True

        # Extract features if required
        if processing_required:
            self.no_fmt_log()
            self.logger.info('Extracting Features')
            file = open(processing_lock,'w')
            file.close()
            self.no_fmt_log()
            self.logger.debug('FX processing lock active')
            self.no_fmt_log()

            pbar = tqdm(desc='Extracting Features ', unit=' batch', colour='blue', total= sum([len(loader) for loader in self.fx_dataloaders.values()]))
            with logging_redirect_tqdm(loggers=[self.logger, self.no_fmt_logger]):
                for split, dataloader in self.fx_dataloaders.items():
                    feature_dir = os.path.join(self.feature_dir,split)
                    os.makedirs(feature_dir,exist_ok=True)
                    for batch in dataloader:
                        input = batch[0].to(self.device)
                        with torch.inference_mode():
                            if self.config.model == 'GE2E':
                                outputs = self.fx_model(input)
                            else:
                                outputs, _ = self.fx_model.extract_features(input)
                                outputs = torch.stack([*outputs],dim=1)
                                    
                        for output, file_name in zip(outputs,batch[1]):
                            file_name = file_name.split('/')[-1].split('.')[0] +'.ftr'
                            torch.save(output,os.path.join(feature_dir,file_name))
                        pbar.update(1)

                time_taken = pbar.format_dict['elapsed']
            pbar.close()

            self.no_fmt_log()
            self.logger.info(f'Time Taken: {pbar.format_interval(time_taken)}')
            self.no_fmt_log()

            os.remove(processing_lock)
            self.logger.info('Feature extraction complete')
            self.no_fmt_log()
            self.logger.debug('FX processing lock released')


    def _get_clf_dataloaders(self):
        
        if self.config.extract_mode=='disk':
            test_dataset = DiskModeClassifierDataset(self.config.model, self.feature_dir, self.dataset_info, 'test')
            train_dataset = DiskModeClassifierDataset(self.config.model, self.feature_dir, self.dataset_info, 'train')
            val_dataset = DiskModeClassifierDataset(self.config.model, self.feature_dir, self.dataset_info, 'validation')
            num_workers = 2
        elif self.config.extract_mode=='memory':
            test_dataset = MemoryModeClassifierDataset(self.config.model, self.fx_model, self.device, self.data_dir, self.dataset_info, 'test')
            train_dataset = MemoryModeClassifierDataset(self.config.model, self.fx_model, self.device, self.data_dir, self.dataset_info, 'train')
            val_dataset = MemoryModeClassifierDataset(self.config.model, self.fx_model, self.device, self.data_dir, self.dataset_info, 'validation')
            num_workers = 4

        train_batch_size = 32
        test_batch_size = 32
        num_workers = 2

        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=num_workers, collate_fn = test_dataset.data_collator)
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size, shuffle=True,
            num_workers=num_workers, collate_fn = train_dataset.data_collator)
        val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=num_workers, collate_fn = val_dataset.data_collator)

        return DotDict(train=train_loader, test=test_loader, validation=val_loader)


    def _train(self, model, dataloader, optimizer, criterion, progress_bar):
        total_loss = 0.0
        model.train()
        for batch in dataloader:
            batch = tuple(input.to(self.device) for input in batch)

            optimizer.zero_grad()
            output = model(batch[0],batch[1])

            loss = criterion(output, batch[2])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            self.acc_metric.update(output, batch[2])

            progress_bar.update(1)
        
        total_loss = total_loss / len(dataloader)
        accuracy = self.acc_metric.compute().item()*100

        self.acc_metric.reset()
        
        return total_loss, accuracy

    
    def _test(self, model, dataloader, criterion, progress_bar):
        total_loss = 0.0
        model.eval()
        for batch in dataloader:
            batch = tuple(input.to(self.device) for input in batch)
           
            with torch.inference_mode():
                output = model(batch[0],batch[1])
                loss = criterion(output, batch[2])

            total_loss += loss.item()
            self.acc_metric.update(output, batch[2])
            
            progress_bar.update(1)
        
        total_loss = total_loss / len(dataloader)
        accuracy = self.acc_metric.compute().item()*100
    
        self.acc_metric.reset()

        return total_loss, accuracy


    def train_pipeline(self):
        
        if self.config.extract_mode=='disk':
            # Extract Features
            self.fx_dataloaders = self._get_fx_dataloaders()
            self._extract_features()
        
        # Setup classifier dataloaders
        self.clf_dataloaders = self._get_clf_dataloaders()

        # Setup
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.clf_model.parameters(), lr=0.001)

        # Set save paths
        best_model_path = os.path.join(self.weights_dir,'best_state.pt')

        train_pbar = tqdm(desc='Training   ', unit=' batch', colour='#42A5F5',total= len(self.clf_dataloaders.train))
        val_pbar   = tqdm(desc='Validation ', unit=' batch', colour='#E0E0E0',total= len(self.clf_dataloaders.validation))
        test_pbar  = tqdm(desc='Testing    ', unit=' batch', colour='#EF5350',total= len(self.clf_dataloaders.test))
        epoch_pbar = tqdm(desc='Epoch      ', unit=' epoch', colour='#43A047',total= self.config.epochs)

        # Train classifier
        self.clf_model.to(self.device)
        best_model = DotDict(val_acc=-1, test_acc=0)
        with logging_redirect_tqdm(loggers=[self.logger, self.no_fmt_logger]):
            self.no_fmt_log()
            self.logger.info('Training Classifier')
            self.no_fmt_log()
            for epoch in range(1,self.config.epochs+1):
                train_loss, train_acc = self._train(self.clf_model, self.clf_dataloaders.train, optimizer, criterion, train_pbar)
                val_loss, val_acc = self._test(self.clf_model, self.clf_dataloaders.validation, criterion, val_pbar)
                test_loss, test_acc = self._test(self.clf_model, self.clf_dataloaders.test, criterion, test_pbar)
                
                if val_acc > best_model.val_acc:
                    best_model.val_acc = val_acc
                    best_model.test_acc = test_acc
                    state = dict(
                        epoch=epoch,
                        val_acc=val_acc,
                        test_acc=test_acc,
                        model_state=self.clf_model.state_dict(),
                        optimizer_state=optimizer.state_dict())
                    torch.save(state,best_model_path)
                    self.logger.debug(f'Best state saved | Val Acc {val_acc} | Test Acc {test_acc}')
                    self.no_fmt_log()
                    
                self.history.train_loss.append(train_loss)
                self.history.train_acc.append(train_acc)
                self.history.val_loss.append(val_loss)
                self.history.val_acc.append(val_acc)
                self.history.test_loss.append(test_loss)
                self.history.test_acc.append(test_acc)

                epoch_pbar.update(1)
                self.logger.info('Epoch: %s |   Train Loss: %.3f | Train Acc: %.2f |   Val Loss: %.3f | Val Acc: %.2f |   Test Loss: %.3f | Tes Acc: %.2f' \
                                %(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

                train_pbar.reset()
                val_pbar.reset()
                test_pbar.reset()

            train_pbar.update(train_pbar.total)
            val_pbar.update(val_pbar.total)
            test_pbar.update(test_pbar.total)
            self.no_fmt_log()
            time_taken = epoch_pbar.format_dict['elapsed']
            train_pbar.close()
            val_pbar.close()
            test_pbar.close()
            epoch_pbar.close()

        self.no_fmt_log()
        self.logger.info(f'Time Taken: {epoch_pbar.format_interval(time_taken)}')
        self.no_fmt_log()

        self.logger.info(f'Best Val Acc : {best_model.val_acc} | Best Test A : {best_model.test_acc}')
        self.no_fmt_log()

        # Agg weights
        if self.config.model != 'GE2E':
            agg_weights = ' '.join([str(weight[0]) for weight in self.clf_model.aggr.state_dict()['weight'][0].detach().cpu().tolist()])
            self.logger.info(f'Agg. Weights : \n{agg_weights}\n')
            agg_weight_path = os.path.join(self.history_dir, 'agg_weights.pt')
            torch.save(agg_weights,agg_weight_path)

        # Save last state
        last_state_path = os.path.join(self.weights_dir, 'last_state.pt')
        state = dict(
            epoch=epoch,
            val_acc=val_acc,
            test_acc=test_acc,
            model_state=self.clf_model.state_dict(),
            optimizer_state=optimizer.state_dict())
        torch.save(state,last_state_path)
        self.logger.debug(f'Last state saved | Val Acc {val_acc} | Test Acc {test_acc}')

        # Save history
        history_path = os.path.join(self.history_dir, 'history.pt')
        torch.save(dict(self.history),history_path)
        self.no_fmt_log()
        self.logger.info(f'History saved ({history_path})')

        return self.history
        

def get_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser("Multilingual SER System")
    parser.add_argument("-r","--run_name", metavar="<str>", default="test",
                        help='Run Name') 
    parser.add_argument("-m","--model", metavar="<str>", default="GE2E", 
                        choices=FX_MODELS, help=str(FX_MODELS))
    parser.add_argument("-em","--extract_mode", metavar="<str>", default="memory",
                        choices=['disk','memory'], help='Disk mode will save the features \
                            to dish and then train, memory mode will process features while training') 
    parser.add_argument("-dv","--device", metavar="<str>", default="gpu",
                        choices=['cpu','gpu'], help='Device to run on') 
    parser.add_argument("-d","--dataset", metavar="<str>", default="EmoDB",
                        choices=DATASETS, help=str(DATASETS))
    
    
    parser.add_argument("-e","--epochs", metavar="<int>", default=20,
                        help='Number of training epochs')

    parser.add_argument("-dd","--data_dir", metavar="<dir>", default="./data",
                        help='Data directory')     
    parser.add_argument("-hd","--history_dir", metavar="<dir>", default="./history",
                        help='History directory')     
    parser.add_argument("-wd","--weights_dir", metavar="<dir>", default="./weights",
                        help='Weights directory')


    parser.add_argument("-ll","--log_level", metavar="<str>", default="info", 
                        choices=list(log_levels.keys()), help=str(list(log_levels.keys())))
    parser.add_argument("-pc","--purge_cache", action="store_true", default=False,
                        help='Purge cached features and extract them again')                                            

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Test
    if args.run_name =='test':
        args.log_level='debug'
        # args.epochs=3
        # args.purge_cache = True
        args.dataset = 'EMOVO'
        args.model = 'WAV2VEC2_BASE'
        args.history_dir = './test'
        os.system(f"rm -rf {os.path.join(args.history_dir,f'v{VERSION}',args.dataset,args.model,args.run_name)}")
        os.system(f"rm -rf {os.path.join(args.weights_dir,f'v{VERSION}',args.dataset,args.model,args.run_name)}")



    # Train Classifier
    trainer = Trainer(args)
    trainer.train_pipeline()