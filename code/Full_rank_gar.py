def full_ranking(model, data, user_item_inter, mask_items, is_training, step, topk, group=False):
    model.eval()
    if mask_items is not None:
        mask_items = torch.LongTensor(list(mask_items)).cuda()
    with torch.no_grad():
        user_emb = model.get_user_emb(model.id_embedding(torch.arange(model.num_user, device='cuda')))
        content = model.feature_extractor()
        item_emb = model.id_embedding(torch.arange(model.num_user, model.num_user + model.num_item, device='cuda'))
        item_emb = model.get_item_emb(content, item_emb)
        result = torch.cat([user_emb, item_emb], dim=0)
        all_index_of_rank_list = rank(model.num_user, user_item_inter, mask_items, result, is_training, step, topk[-1])
        gt_list = [None for _ in range(model.num_user)]
        for u_id in data:
            gt_list[u_id] = data[u_id]
        if not is_training:
            for u_id in range(model.num_user):
                if gt_list[u_id] is not None:
                    print(f"User {u_id}: Top-10 predicted: {all_index_of_rank_list[u_id][:10]}, Ground truth: {gt_list[u_id]}")
                    break
        if group:
            results = group_computeTopNAccuracy(gt_list, all_index_of_rank_list, group, topk)
        else:
            results = computeTopNAccuracy(gt_list, all_index_of_rank_list, topk)
    return results
