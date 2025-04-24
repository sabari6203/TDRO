from gar_model import GARModel  # Assuming the GAR model is saved as model_gar.py
from Dataset import data_load  # Use your updated data loading function
import argparse

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='kwai_small', help='Dataset path')
    parser.add_argument('--model_name', default='SSL', help='Model Name.')
    parser.add_argument('--log_name', default='', help='log name.')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    parser.add_argument("--topK", default='[10, 20, 50, 100]', help="the recommended item num")
    parser.add_argument('--step', type=int, default=2000, help='Workers number.')

    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--dim_E', type=int, default=128, help='Embedding dimension.')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='Weight decay.')

    # CLCRec
    parser.add_argument('--num_neg', type=int, default=256, help='Negative size.')
    parser.add_argument('--contrastive', type=float, default=0.1, help='Weight loss one.')
    parser.add_argument('--num_sample', type=float, default=0.5, help='probability of robust training.')
    parser.add_argument('--temp_value', type=float, default=0.1, help='Contrastive temp_value.')
    
    parser.add_argument('--pretrained_emb', type=str, default='./pretrained_emb/', help='path of pretrained embedding of items')

    # Group-DRO
    parser.add_argument('--num_group', type=int, default=1, help='group number for group DRO')
    parser.add_argument('--mu', type=float, default=0.5, help='streaming learning rate for group DRO')
    parser.add_argument('--eta', type=float, default=0.01, help='step size for group DRO')

    # TDRO
    parser.add_argument('--num_period', type=int, default=1, help='time period number for TDRO')
    parser.add_argument('--split_mode', type=str, default='global', help='split the group by global time or relative interactions per user', choices=['relative','global'])
    parser.add_argument('--lam', type=float, default=0.5, help='coefficient for time-aware shifting trend')
    parser.add_argument('--p', type=float, default=0.2, help='strength of gradient (see more in common good)')

    # group evaluation
    parser.add_argument('--group_test', action='store_true', help='whether or not do evaluation of user/item groups')
    parser.add_argument('--portion_list', type=str, help='portion list of different groups')

    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--save_path', default='./models/', help='model save path')
    parser.add_argument('--inference',action='store_true', help='only inference stage')
    parser.add_argument('--ckpt', type=str, help='pretrained model path')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = init()  # Parsing arguments
    
    # Load data
    num_user, num_item, train_data, val_data, test_data, v_feat, a_feat, t_feat = data_load(args.data_path)

    # Instantiate your GAR model
    model = GARModel(num_user, num_item, v_feat, a_feat, t_feat, args.dim_E).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.l_r, weight_decay=args.reg_weight)

    # Training loop
    for epoch in range(args.num_epoch):
        train_loss = train_GAR(train_data, model, optimizer, args)
        print(f"Epoch {epoch}, Loss: {train_loss}")
        
        # Validation and Testing
        if epoch % 5 == 0:  # Check validation/test every few epochs
            val_results = full_ranking(model, val_data, None)  # Example, customize as needed
            test_results = full_ranking(model, test_data, None)
            print("Validation Results:", val_results)
            print("Test Results:", test_results)
