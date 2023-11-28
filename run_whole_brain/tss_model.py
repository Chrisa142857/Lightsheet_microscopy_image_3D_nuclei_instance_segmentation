import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GCNConv, GATConv
import os.path as osp
import os
from torch import nn
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_geometric.loader import DataLoader

NODE_FEAT_SIZE = 209
EDGE_IN_DIM = 6
Message_Passing_Step = 2


def get_model_config():
    # graph in z dim
    drop_out = 0.3
    edge_fc_dims = [32, 64, 128, 256] # big encoder 2
    edge_out_dim = 256 # big MPN 2
    edge_model_fc_dims = [1024, 512, 512, 256] # big 2
    node_fc_dims = [256, 512, 512, 1024]
    node_out_dim = 1024
    node_model_fc_dims = [1024, 512]
    # edge_fc_dims = [16, 32] # big encoder 1
    # edge_out_dim = 32 # big MPN 1
    # edge_model_fc_dims = [128, 64, 32] # big 1
    # node_fc_dims = [64, 32]
    # node_out_dim = 64
    # node_model_fc_dims = [128, 64]
    model_config = { # default config is in https://github.com/dvl-tum/mot_neural_solver/blob/master/configs/tracking_cfg.yaml
        'node_agg_fn': 'sum',
        'num_enc_steps': Message_Passing_Step,  # Number of message passing steps
        'num_class_steps': Message_Passing_Step - 1,  # Number of message passing steps during feature vectors are classified (after Message Passing)
        'reattach_initial_nodes': False,  # Determines whether initially encoded node feats are used during node updates
        'reattach_initial_edges': True,  # Determines whether initially encoded edge feats are used during node updates
        'encoder_feats_dict':{
            # 'edge_in_dim': EDGE_IN_DIM,
            'edge_in_dim': EDGE_IN_DIM,
            # 'edge_fc_dims': [18, 18],
            # 'edge_out_dim': 16, # default
            'edge_fc_dims': edge_fc_dims, # big encoder 1
            'edge_out_dim': edge_out_dim, # big MPN 1
            'node_in_dim': NODE_FEAT_SIZE,
            'node_fc_dims': node_fc_dims, # big encoder 1
            'node_out_dim': node_out_dim, # big encoder 1
            'dropout_p': drop_out,
            'use_batchnorm': True,
            # 'dropout_p': 0, # default
            # 'use_batchnorm': False,
        },
        # In size is 4 * encoded nodes + 2 * encoded edges
        'edge_model_feats_dict': { # the last fc out dim == encoder_feats_dict.edge_out_dim
            # 'fc_dims': [80, 32], # default
            'fc_dims': edge_model_fc_dims,
            'dropout_p': drop_out,
            'use_batchnorm': True,
            # 'dropout_p': 0, # default
            # 'use_batchnorm': False,
        },
        # In size is 2 * encoded nodes + 1 * encoded edges
        'node_model_feats_dict':{
            # 'fc_dims': [56, 32],
            'fc_dims': node_model_fc_dims,
            'dropout_p': drop_out,
            'use_batchnorm': True,
            # 'dropout_p': 0, # default
            # 'use_batchnorm': False,
        },
        'classifier_feats_dict':{
            'edge_in_dim': edge_out_dim, # edge_in_dim == encoder_feats_dict.edge_out_dim
            'edge_fc_dims': [256, 512, 256],
            'edge_out_dim': 256,
            'dropout_p': drop_out,
            'use_batchnorm': True,
            # 'dropout_p': 0, # default
            # 'use_batchnorm': False,
        },
        'cnn_params':{
            'arch': 'resnet50',
            'model_weights_path':{
                'resnet50': 'trained_models/reid/resnet50_market_cuhk_duke.tar-232',
            },
        }
    }
    return model_config

class MLP(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False):
        super(MLP, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1:
                layers.append(nn.LeakyReLU(inplace=True))

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_layers(input)


class MetaLayer(torch.nn.Module):
    """
    Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
    (https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html)
    """
    def __init__(self, edge_model=None, node_model=None):
        """
        Args:
            edge_model: Callable Edge Update Model
            node_model: Callable Node Update Model
        """
        super(MetaLayer, self).__init__()

        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)

        Returns: Updated Node and Edge Feature matrices

        """
        # row, col = edge_index

        # Edge Update
        if self.edge_model is not None:
            edge_attr = self.edge_model(x[edge_index[0]], x[edge_index[1]], edge_attr)

        # Node Update
        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr)

        return x, edge_attr

    def __repr__(self):
        return '{}(edge_model={}, node_model={})'.format(self.__class__.__name__, self.edge_model, self.node_model)

class EdgeModel(nn.Module):
    """
    Class used to peform the edge update during Neural message passing
    """
    def __init__(self, edge_mlp):
        super(EdgeModel, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, source, target, edge_attr):
        out = torch.cat([source, target, edge_attr], dim=1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """
    def __init__(self, flow_in_mlp, flow_out_mlp, node_mlp, node_agg_fn):
        super(NodeModel, self).__init__()

        self.flow_in_mlp = flow_in_mlp
        self.flow_out_mlp = flow_out_mlp
        self.node_mlp = node_mlp
        self.node_agg_fn = node_agg_fn

    def forward(self, x, edge_index, edge_attr):
        # row, col = edge_index
        flow_out_mask = edge_index[0] < edge_index[1]
        flow_out_row, flow_out_col = edge_index[0][flow_out_mask], edge_index[1][flow_out_mask]
        flow_out_input = torch.cat([x[flow_out_col], edge_attr[flow_out_mask]], dim=1)
        flow_out = self.flow_out_mlp(flow_out_input)
        flow_out = self.node_agg_fn(flow_out, flow_out_row, x.size(0))

        flow_in_mask = edge_index[0] > edge_index[1]
        flow_in_row, flow_in_col = edge_index[0][flow_in_mask], edge_index[1][flow_in_mask]
        flow_in_input = torch.cat([x[flow_in_col], edge_attr[flow_in_mask]], dim=1)
        flow_in = self.flow_in_mlp(flow_in_input)

        flow_in = self.node_agg_fn(flow_in, flow_in_row, x.size(0))
        flow = torch.cat((flow_in, flow_out), dim=1)

        return self.node_mlp(flow)

class MLPGraphIndependent(nn.Module):
    """
    Class used to to encode (resp. classify) features before (resp. after) neural message passing.
    It consists of two MLPs, one for nodes and one for edges, and they are applied independently to node and edge
    features, respectively.

    This class is based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """

    def __init__(self, edge_in_dim = None, node_in_dim = None, edge_out_dim = None, node_out_dim = None,
                 node_fc_dims = None, edge_fc_dims = None, dropout_p = None, use_batchnorm = None):
        super(MLPGraphIndependent, self).__init__()

        if node_in_dim is not None :
            self.node_mlp = MLP(input_dim=node_in_dim, fc_dims=list(node_fc_dims) + [node_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.node_mlp = None

        if edge_in_dim is not None :
            self.edge_mlp = MLP(input_dim=edge_in_dim, fc_dims=list(edge_fc_dims) + [edge_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.edge_mlp = None

    def forward(self, edge_feats = None, nodes_feats = None):

        if self.node_mlp is not None:
            out_node_feats = self.node_mlp(nodes_feats)

        else:
            out_node_feats = nodes_feats

        if self.edge_mlp is not None:
            out_edge_feats = self.edge_mlp(edge_feats)

        else:
            out_edge_feats = edge_feats

        return out_edge_feats, out_node_feats

class TSS(nn.Module):
    """
    Main Model Class. Contains all the components of the model. It consists of of several networks:
    - 2 encoder MLPs (1 for nodes, 1 for edges) that provide the initial node and edge embeddings, respectively,
    - 4 update MLPs (3 for nodes, 1 per edges used in the 'core' Message Passing Network
    - 1 edge classifier MLP that performs binary classification over the Message Passing Network's output.

    This class was initially based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """

    def __init__(self, model_params = get_model_config(), bb_encoder = None):
        """
        Defines all components of the model
        Args:
            bb_encoder: (might be 'None') CNN used to encode bounding box apperance information.
            model_params: dictionary contaning all model hyperparameters
        """
        super(TSS, self).__init__()

        self.node_cnn = bb_encoder
        self.model_params = model_params

        # Define Encoder and Classifier Networks
        encoder_feats_dict = model_params['encoder_feats_dict']
        classifier_feats_dict = model_params['classifier_feats_dict']

        self.encoder = MLPGraphIndependent(**encoder_feats_dict)
        self.classifier = MLPGraphIndependent(**classifier_feats_dict)

        # Define the 'Core' message passing network (i.e. node and edge update models)
        self.MPNet = self._build_core_MPNet(model_params=model_params, encoder_feats_dict=encoder_feats_dict)

        self.num_enc_steps = model_params['num_enc_steps']
        self.num_class_steps = model_params['num_class_steps']


    def _build_core_MPNet(self, model_params, encoder_feats_dict):
        """
        Builds the core part of the Message Passing Network: Node Update and Edge Update models.
        Args:
            model_params: dictionary contaning all model hyperparameters
            encoder_feats_dict: dictionary containing the hyperparameters for the initial node/edge encoder
        """

        # Define an aggregation operator for nodes to 'gather' messages from incident edges
        node_agg_fn = model_params['node_agg_fn']
        assert node_agg_fn.lower() in ('mean', 'max', 'sum'), "node_agg_fn can only be 'max', 'mean' or 'sum'."

        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size)

        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]

        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)

        # Define all MLPs involved in the graph network
        # For both nodes and edges, the initial encoded features (i.e. output of self.encoder) can either be
        # reattached or not after each Message Passing Step. This affects MLPs input dimensions
        self.reattach_initial_nodes = model_params['reattach_initial_nodes']
        self.reattach_initial_edges = model_params['reattach_initial_edges']

        edge_factor = 2 if self.reattach_initial_edges else 1
        node_factor = 2 if self.reattach_initial_nodes else 1

        edge_model_in_dim = node_factor * 2 * encoder_feats_dict['node_out_dim'] + edge_factor * encoder_feats_dict[
            'edge_out_dim']
        # edge_model_in_dim = node_factor * 2 * encoder_feats_dict['node_out_dim'] + edge_factor * model_params[
        #     'edge_model_feats_dict']['fc_dims'][-1]
        # node_model_in_dim = node_factor * encoder_feats_dict['node_out_dim'] + encoder_feats_dict['edge_out_dim'] 
        node_model_in_dim = node_factor * encoder_feats_dict['node_out_dim'] + model_params['edge_model_feats_dict']['fc_dims'][-1]

        # Define all MLPs used within the MPN
        edge_model_feats_dict = model_params['edge_model_feats_dict']
        node_model_feats_dict = model_params['node_model_feats_dict']

        edge_mlp = MLP(input_dim=edge_model_in_dim,
                       fc_dims=edge_model_feats_dict['fc_dims'],
                       dropout_p=edge_model_feats_dict['dropout_p'],
                       use_batchnorm=edge_model_feats_dict['use_batchnorm'])

        flow_in_mlp = MLP(input_dim=node_model_in_dim,
                          fc_dims=node_model_feats_dict['fc_dims'],
                          dropout_p=node_model_feats_dict['dropout_p'],
                          use_batchnorm=node_model_feats_dict['use_batchnorm'])

        flow_out_mlp = MLP(input_dim=node_model_in_dim,
                           fc_dims=node_model_feats_dict['fc_dims'],
                           dropout_p=node_model_feats_dict['dropout_p'],
                           use_batchnorm=node_model_feats_dict['use_batchnorm'])

        # node_mlp = nn.Sequential(*[nn.Linear(2 * encoder_feats_dict['node_out_dim'], encoder_feats_dict['node_out_dim']),
        #                            nn.LeakyReLU(inplace=True)])
        node_mlp = nn.Sequential(*[nn.Linear(2 * model_params['node_model_feats_dict']['fc_dims'][-1], encoder_feats_dict['node_out_dim']),
                                   nn.LeakyReLU(inplace=True)])

        # Define all MLPs used within the MPN
        return MetaLayer(edge_model=EdgeModel(edge_mlp = edge_mlp),
                         node_model=NodeModel(flow_in_mlp = flow_in_mlp,
                                                       flow_out_mlp = flow_out_mlp,
                                                       node_mlp = node_mlp,
                                                       node_agg_fn = node_agg_fn))


    def forward(self, x, edge_index, edge_attr):
        """
        Provides a fractional solution to the data association problem.
        First, node and edge features are independently encoded by the encoder network. Then, they are iteratively
        'combined' for a fixed number of steps via the Message Passing Network (self.MPNet). Finally, they are
        classified independently by the classifiernetwork.
        Args:
            data: object containing attribues
              - x: node features matrix
              - edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
                graph adjacency (i.e. edges) (i.e. sparse adjacency)
              - edge_attr: edge features matrix (sorted by edge apperance in edge_index)

        Returns:
            classified_edges: list of unnormalized node probabilites after each MP step
        """
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x_is_img = len(x.shape) == 4
        if self.node_cnn is not None and x_is_img:
            x = self.node_cnn(x)

            emb_dists = nn.functional.pairwise_distance(x[edge_index[0]], x[edge_index[1]]).view(-1, 1)
            edge_attr = torch.cat((edge_attr, emb_dists), dim = 1)

        # Encoding features step
        latent_edge_feats, latent_node_feats = self.encoder(edge_attr, x)
        initial_edge_feats = latent_edge_feats
        initial_node_feats = latent_node_feats

        # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
        # passing steps are classified in order to compute the loss.
        first_class_step = self.num_enc_steps - self.num_class_steps + 1
        outputs_dict = {'classified_edges': []}
        for step in range(1, self.num_enc_steps + 1):

            # Reattach the initially encoded embeddings before the update
            if self.reattach_initial_edges:
                latent_edge_feats = torch.cat((initial_edge_feats, latent_edge_feats), dim=1)
            if self.reattach_initial_nodes:
                latent_node_feats = torch.cat((initial_node_feats, latent_node_feats), dim=1)

            # Message Passing Step
            latent_node_feats, latent_edge_feats = self.MPNet(latent_node_feats, edge_index, latent_edge_feats)

            if step >= first_class_step:
                # Classification Step
                dec_edge_feats, _ = self.classifier(latent_edge_feats)
                outputs_dict['classified_edges'].append(dec_edge_feats)

        if self.num_enc_steps == 0:
            dec_edge_feats, _ = self.classifier(latent_edge_feats)
            outputs_dict['classified_edges'].append(dec_edge_feats)
        
        return outputs_dict['classified_edges']
