'''Modules for hypernetwork experiments, Paper Sec. 4.4
'''

import torch
from torch import nn
from collections import OrderedDict
import omni.isaac.orbit_envs.soft.Oring_entangled.models.modules as modules
import omni.isaac.orbit_envs.soft.Oring_entangled.models.sdf_meshing as sdf_meshing


class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = modules.FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                                 num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
                                 outermost_linear=True, nonlinearity='relu')
            self.nets.append(hn)

            if 'weight' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params


class NeuralProcessImplicit2DHypernet(nn.Module):
    '''A canonical 2D representation hypernetwork mapping 2D coords to out_features.'''
    def __init__(self, in_features, out_features, image_resolution=None, encoder_nl='sine'):
        super().__init__()

        latent_dim = 256
        self.hypo_net = modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution,
                                             in_features=2)
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=256,
                                      hypo_module=self.hypo_net)
        self.set_encoder = modules.SetEncoder(in_features=in_features, out_features=latent_dim, num_hidden_layers=2,
                                              hidden_features=latent_dim, nonlinearity=encoder_nl)
        print(self)

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False

    def get_hypo_net_weights(self, model_input):
        pixels, coords = model_input['img_sub'], model_input['coords_sub']
        ctxt_mask = model_input.get('ctxt_mask', None)
        embedding = self.set_encoder(coords, pixels, ctxt_mask=ctxt_mask)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            pixels, coords = model_input['img_sub'], model_input['coords_sub']
            ctxt_mask = model_input.get('ctxt_mask', None)
            embedding = self.set_encoder(coords, pixels, ctxt_mask=ctxt_mask)
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        model_output = self.hypo_net(model_input, params=hypo_params)
        return {'model_in':model_output['model_in'], 'model_out':model_output['model_out'], 'latent_vec':embedding,
                'hypo_params':hypo_params}


class ConvolutionalNeuralProcessImplicit2DHypernet(nn.Module):
    def __init__(self, in_features, out_features, image_resolution=None, partial_conv=False):
        super().__init__()
        latent_dim = 256

        if partial_conv:
            self.encoder = modules.PartialConvImgEncoder(channel=in_features, image_resolution=image_resolution)
        else:
            self.encoder = modules.ConvImgEncoder(channel=in_features, image_resolution=image_resolution)
        self.hypo_net = modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution,
                                             in_features=2)
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=256,
                                      hypo_module=self.hypo_net)
        print(self)

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            embedding = self.encoder(model_input['img_sparse'])
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        model_output = self.hypo_net(model_input, params=hypo_params)

        return {'model_in': model_output['model_in'], 'model_out': model_output['model_out'], 'latent_vec': embedding,
                'hypo_params': hypo_params}

    def get_hypo_net_weights(self, model_input):
        embedding = self.encoder(model_input['img_sparse'])
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

class PCNNeuralProcessImplicit3DHypernet(nn.Module):
    def __init__(self, type='sine', is_vae=False, latent_skip=False):
        super().__init__()
        self.latent_dim = 64

        self.is_vae = is_vae

        self.hypo_net =  modules.SingleBVPNet(type=type, in_features=3, hidden_features=64, num_hidden_layers=3, latent_in=self.latent_dim if latent_skip else 0) # sine or relu
        self.encoder = modules.PCNEncoder(latent_dim=self.latent_dim, is_vae=is_vae)
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=3, hyper_hidden_features=128, hypo_module=self.hypo_net)
        
        print(self)

    def forward(self, model_input):
        if model_input.get('partial_embedding', None) is None:
            partial_mu, partial_logvar = self.encoder(model_input['partial'])
        else:
            partial_mu = model_input['partial_mu']
            partial_logvar = model_input['partial_logvar']
            

        if model_input.get('complete_embedding', None) is None:
            complete_mu, complete_logvar = self.encoder(model_input['complete'])
        else:
            complete_mu = model_input['partial_mu']
            complete_logvar = model_input['partial_logvar']
        
        if self.is_vae and self.training:
            partial_embedding = self.reparameterize(partial_mu, partial_logvar)
            complete_embedding = self.reparameterize(complete_mu, complete_logvar)
        else:
            partial_embedding = partial_mu
            complete_embedding = complete_mu
        
        model_input['partial_embedding'] = partial_embedding
        model_input['complete_embedding'] = complete_embedding

        # hypo_params = self.hyper_net(partial_embedding)
        hypo_params = self.hyper_net(complete_embedding)
        
        model_output = self.hypo_net(model_input, params=hypo_params)

        return {'model_in': model_output['model_in'], 'model_out': model_output['model_out'], 'partial_latent_vec': partial_embedding, 'partial_mu': partial_mu, 'partial_logvar': partial_logvar, 'hypo_params': hypo_params, 'complete_latent_vec': complete_embedding, 'complete_mu': complete_mu, 'complete_logvar':complete_logvar}

    def encode(self, xyz):
        mu, logvar = self.encoder(xyz) # new axis for batch?
        if self.is_vae and self.training:
            embedding = self.reparameterize(mu, logvar)
        else:
            embedding = mu
        return embedding
    
    def get_hypo_net_weights(self, model_input):
        embedding, _ = self.encoder(model_input['partial'])
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

class NeuralProcessImplicit3DHypernet(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = model_cfg
        self.latent_dim = model_cfg['latent_dim']
        self.is_vae = model_cfg['vae']
        self.encoder_type = model_cfg['encoder']['type']
        self.skip = model_cfg['skip']

        self.hypo_net =  modules.SingleBVPNet(type=self.cfg['hyponet']['type'], in_features=3, hidden_features=self.cfg['hyponet']['hidden_features'], num_hidden_layers=self.cfg['hyponet']['hidden_layers'], latent_in=self.latent_dim if self.skip else 0) # sine or relu
        if self.encoder_type == 'pcn': 
            self.encoder = modules.PCNEncoder(latent_dim=self.latent_dim, is_vae=self.is_vae)
        elif self.encoder_type == 'pointnet':
            self.encoder = modules.PointNetPPEncoder(latent_dim = self.latent_dim, normal_channel=False, is_vae=self.is_vae)
        else:
            raise NotImplementedError
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=self.cfg['decoder']['hidden_layers'], hyper_hidden_features=self.cfg['decoder']['hidden_features'], hypo_module=self.hypo_net)
        
        print(self)

    def forward(self, model_input):
        if model_input.get('partial_embedding', None) is None:
            partial_mu, partial_logvar = self.encoder(model_input['partial'])
        else:
            partial_mu = model_input['partial_mu']
            partial_logvar = model_input['partial_logvar']
            

        if model_input.get('complete_embedding', None) is None:
            complete_mu, complete_logvar = self.encoder(model_input['complete'])
        else:
            complete_mu = model_input['partial_mu']
            complete_logvar = model_input['partial_logvar']
        
        if self.is_vae and self.training:
            partial_embedding = self.reparameterize(partial_mu, partial_logvar)
            complete_embedding = self.reparameterize(complete_mu, complete_logvar)
        else:
            partial_embedding = partial_mu
            complete_embedding = complete_mu
        
        model_input['partial_latent_vec'] = partial_embedding
        model_input['complete_latent_vec'] = complete_embedding

        hypo_params = self.hyper_net(partial_embedding)
        # hypo_params = self.hyper_net(complete_embedding)
        
        model_output = self.hypo_net(model_input, params=hypo_params)

        return {'model_in': model_output['model_in'], 'model_out': model_output['model_out'], 'partial_latent_vec': partial_embedding, 'partial_mu': partial_mu, 'partial_logvar': partial_logvar, 'hypo_params': hypo_params, 'complete_latent_vec': complete_embedding, 'complete_mu': complete_mu, 'complete_logvar':complete_logvar}


    def encode(self, xyz):
        mu, logvar = self.encoder(xyz) # new axis for batch?
        if self.is_vae and self.training:
            embedding = self.reparameterize(mu, logvar)
        else:
            embedding = mu
        return embedding
    
    def get_hypo_net_weights(self, model_input):
        embedding, _ = self.encoder(model_input['partial'])
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std






class PointNetPPNeuralProcessImplicit3DHypernet(nn.Module):
    def __init__(self, type='sine', is_vae=False, latent_skip=False):
        super().__init__()
        self.latent_dim = 128
        self.is_vae = is_vae

        self.hypo_net =  modules.SingleBVPNet(type=type, in_features=3, hidden_features=64, num_hidden_layers=3, latent_in=self.latent_dim if latent_skip else 0) # sine or relu
        self.encoder = modules.PointNetPPEncoder(normal_channel=False, is_vae=is_vae)
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=3, hyper_hidden_features=128, hypo_module=self.hypo_net)
        
        
        print(self)

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            mu, logvar = self.encoder(model_input['partial'])
        else:
            mu = model_input['embedding']
            logvar = model_input['logvar']
            

        if model_input.get('complete', None) is None:
            complete_embedding = None
        else:
            complete_mu, complete_logvar = self.encoder(model_input['complete'])
        
        if self.is_vae and self.training:
            embedding = self.reparameterize(mu, logvar)
            complete_embedding = complete_mu
        else:
            embedding = mu
            complete_embedding = complete_mu
        
        model_input['embedding'] = embedding

        hypo_params = self.hyper_net(embedding)
        
        model_output = self.hypo_net(model_input, params=hypo_params)

        return {'model_in': model_output['model_in'], 'model_out': model_output['model_out'], 'latent_vec': embedding, 'mu': mu, 'logvar': logvar, 'hypo_params': hypo_params, 'complete_latent_vec': complete_embedding, 'complete_mu': complete_mu, 'complete_logvar':complete_logvar}

    def encode(self, xyz):
        mu, logvar = self.encoder(xyz) # new axis for batch?
        if self.is_vae and self.training:
            embedding = self.reparameterize(mu, logvar)
        else:
            embedding = mu
        return embedding
    
    def get_hypo_net_weights(self, model_input):
        embedding, _ = self.encoder(model_input['partial'])
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class PointCloudHypernet(nn.Module):
    def __init(self, in_features, out_features, partial_conv=False, hypo_type='sine'):
        super().__init__()
        latent_dim = 256

        self.hypo_net = modules.SingleBVPNet(out_features=out_features, type=hypo_type, in_features=3)
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=256, hypo_module=self.hypo_net)

        print(self)

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            embedding = self.encoder(model_input['img_sparse'])
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        model_output = self.hypo_net(model_input, params=hypo_params)

        return {'model_in': model_output['model_in'], 'model_out': model_output['model_out'], 'latent_vec': embedding,
                'hypo_params': hypo_params}

    def get_hypo_net_weights(self, model_input):
        embedding = self.encoder(model_input['img_sparse'])
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False


class SDFDecoder(torch.nn.Module):
    def __init__(self, model=None):
        super().__init__()
        # Define the model.
        if model is None:
            self.model = PCNNeuralProcessImplicit3DHypernet()
            self.model.cuda()
        else:
            self.model = model
        
    def forward(self, coords):
        with torch.no_grad():
            hypo_model_in = {'coords': coords, 'partial_latent_vec': self.embedding}
            out = self.model.hypo_net(hypo_model_in, params=self.hypo_params)['model_out']
        return out
    
    def decode(self, embedding):
        with torch.no_grad():
            self.eval()
            self.hypo_params = self.model.hyper_net(embedding)
            self.embedding = embedding
            pcd = sdf_meshing.create_pcd(self, N=64)
        return pcd

############################
# Initialization schemes
def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1/fan_in, 1/fan_in)
