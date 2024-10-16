from network.gps_model import GPSModel
from polynormer import Polynormer
from sgformer import SGFormer, GCN
from nodeformer import NodeFormer
from network.multi_model import MultiModel
from nagphormer import TransformerModel
from goat import Transformer
def parse_method(args, n, c, d, device):
    if args.model == "GPS":
        model = GPSModel(args, d, c).to(device)
    elif args.model =="polynormer":
        model = Polynormer(d, args.hidden_channels, c, local_layers=args.local_layers, global_layers=args.global_layers,
            in_dropout=args.in_dropout, dropout=args.dropout, global_dropout=args.global_dropout,
            heads=args.num_heads, beta=args.beta, pre_ln=args.pre_ln).to(device)
    elif args.model == 'nodeformer':
        model = NodeFormer(in_channels=d,
                         hidden_channels=args.hidden_channels,
                         out_channels=c,
                         num_layers=args.global_layers,
                         dropout=args.dropout,
                         num_heads=args.num_heads,
                         use_bn=args.use_bn).to(device)
    elif args.model == 'sgformer':
        gnn = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=args.hidden_channels,
                    num_layers=args.layers,
                    dropout=args.dropout)
        model = SGFormer(d, args.hidden_channels, c, num_layers=args.layers, alpha=args.alpha, dropout=args.dropout, 
                         num_heads=args.num_heads, use_bn=args.use_bn, use_residual=args.use_residual, 
                         use_graph=args.use_graph, use_weight=args.use_weight, use_act=args.use_act, 
                         graph_weight=args.graph_weight, gnn=gnn, aggregate=args.aggregate, jk=args.jk).to(device)
    elif args.model == 'exphormer':
        model = MultiModel(args, d, c).to(device)
    elif args.model == 'nagphormer':
        model = TransformerModel(hops=args.hops, 
                        n_class=c, 
                        input_dim=d, 
                        pe_dim = args.hidden_channels,
                        n_layers=args.global_layers,
                        num_heads=args.num_heads,
                        hidden_dim=args.hidden_channels,
                        ffn_dim=args.hidden_channels,
                        dropout_rate=args.dropout,
                        attention_dropout_rate=args.dropout).to(device)
    else:
        model = Transformer(
            num_nodes=n,
            in_channels=d,
            hidden_channels=args.hidden_channels, 
            out_channels=c,
            global_dim=args.hidden_channels,
            num_layers=args.layers,
            heads=args.num_heads,
            ff_dropout=args.dropout,
            attn_dropout=args.dropout,
            skip=0,
            dist_count_norm=1,
            conv_type='full',
            num_centroids=args.num_centroids,
            no_bn=False,
            norm_type='batch_norm'
        ).to(device)
    
    return model
        

def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')
    parser.add_argument('--model', type=str, default='MPNN')
    # GNN
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--pre_linear', action='store_true')
    parser.add_argument('--res', action='store_true', help='use residual connections for GNNs')
    parser.add_argument('--ln', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--bn', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--jk', action='store_true', help='use JK for GNNs')
    # GPS
    parser.add_argument('--pe_types', type=str, default='RWSE')
    parser.add_argument('--local_gnn_type', type=str, default='GCN')
    parser.add_argument('--layers', type=int, default=1)
    # poly
    parser.add_argument('--global_layers', type=int, default=2,
                        help='number of layers for global attention')
    parser.add_argument('--beta', type=float, default=-1.0,
                        help='Polynormer beta initialization')
    parser.add_argument('--global_dropout', type=float, default=None)
    # sgformer
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weight for residual link')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_graph', action='store_true', help='use pos emb')
    parser.add_argument('--use_weight', action='store_true')
    parser.add_argument('--attention', type=str, default='gcn')
    parser.add_argument('--graph_weight', type=float,
                        default=0.8, help='graph weight.')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each trans layer')
    parser.add_argument('--use_act', action='store_true', help='use activation for each trans layer')
    parser.add_argument('--aggregate', type=str, default='add',
                        help='aggregate type, add or cat.')
    # nagphormer
    parser.add_argument('--hops', type=int, default=10)
    # goat
    parser.add_argument('--num_centroids', type=int, default=4096)

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--in_dropout', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.5)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')
    parser.add_argument('--save_result', action='store_true', help='whether to save result')


