from model_gar import GARModel  # Assuming the GAR model is saved as model_gar.py
from Dataset import data_load  # Use your updated data loading function

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
