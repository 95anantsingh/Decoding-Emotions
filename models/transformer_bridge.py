
import torch.nn as nn
from collections import OrderedDict

class TransformerBridge(nn.Module):
    def __init__(self, fx_model, clf_model, layer):
        super().__init__()
    
        # Freeze all the parameters
        for param in fx_model.parameters():
            param.requires_grad = False
        
        try: 
            # For zeroth layer
            if layer==0:
                # Unfreeze the parameters for the given layer
                for param in fx_model.encoder.transformer.pos_conv_embed.parameters():
                    param.requires_grad = True
                # Slice the model
                self.fx =   nn.Sequential(OrderedDict([
                            ('feature_extractor', nn.Sequential(OrderedDict(dict(fx_model.feature_extractor.named_children())))),
                            ('encoder', nn.Sequential(
                                ('feature_projection',OrderedDict(dict(fx_model.encoder.feature_projection.named_children()))),
                                ('transformer',nn.Sequential(
                                    ('pos_conv_embed',OrderedDict(dict(fx_model.encoder.transformer.pos_conv_embed.named_children())))
                                ))         
                            ))
                       ]))
            # For nth layer
            else:  
                # Unfreeze the parameters for the given layer
                for param in fx_model.encoder.transformer.layers[layer-1].parameters():
                    param.requires_grad = True
                # Slice the model
                self.fx =   nn.Sequential(OrderedDict([
                                ('feature_extractor', nn.Sequential(OrderedDict(dict(fx_model.feature_extractor.named_children())))),
                                ('encoder', nn.Sequential(
                                    ('feature_projection',OrderedDict(dict(fx_model.encoder.feature_projection.named_children()))),
                                    ('transformer',nn.Sequential(
                                        ('pos_conv_embed',OrderedDict(dict(fx_model.encoder.transformer.pos_conv_embed.named_children()))),
                                        ('layer_norm',fx_model.encoder.transformer.layer_norm),
                                        ('dropout',fx_model.encoder.transformer.dropout),
                                        ('layers',nn.Sequential(OrderedDict(dict(fx_model.encoder.layers[:layer].named_children()))))
                                    ))         
                                ))
                        ]))
        except: 
            # For zeroth layer
            if layer==0:
                # Unfreeze the parameters for the given layer
                for param in fx_model.model.encoder.transformer.pos_conv_embed.parameters():
                    param.requires_grad = True
                # Slice the model
                self.fx =   nn.Sequential(OrderedDict([
                            ('feature_extractor', nn.Sequential(OrderedDict(dict(fx_model.model.feature_extractor.named_children())))),
                            ('encoder', nn.Sequential(
                                ('feature_projection',OrderedDict(dict(fx_model.model.encoder.feature_projection.named_children()))),
                                ('transformer',nn.Sequential(
                                    ('pos_conv_embed',OrderedDict(dict(fx_model.model.encoder.transformer.pos_conv_embed.named_children())))
                                ))         
                            ))
                    ]))
            # For nth layer
            else:
                # Unfreeze the parameters for the given layer
                for param in fx_model.model.encoder.transformer.layers[layer-1].parameters():
                    param.requires_grad = True
                # Slice the model
                self.fx =   nn.Sequential(OrderedDict([
                                ('feature_extractor', nn.Sequential(OrderedDict(dict(fx_model.model.feature_extractor.named_children())))),
                                ('encoder', nn.Sequential(
                                    ('feature_projection',OrderedDict(dict(fx_model.model.encoder.feature_projection.named_children()))),
                                    ('transformer',nn.Sequential(
                                        ('pos_conv_embed',OrderedDict(dict(fx_model.model.encoder.transformer.pos_conv_embed.named_children()))),
                                        ('layer_norm',fx_model.model.encoder.transformer.layer_norm),
                                        ('dropout',fx_model.model.encoder.transformer.dropout),
                                        ('layers',nn.Sequential(OrderedDict(dict(fx_model.model.encoder.layers[:layer].named_children()))))
                                    ))         
                                ))
                        ]))

        self.clf = clf_model

    def forward(self, x):
        features = self.fx(x)
        logits = self.clf(features)
        return logits