import torch
import random
import gc
import random
from random import randrange
import warnings

def GetModifyIndex(image_feature_len, trigger_feature_len, position_i=None):
    assert image_feature_len > trigger_feature_len
    if position_i == 'mid':
        rt = int(image_feature_len/2)
    elif position_i == 'start':
        rt = 0
    elif position_i == 'end':
        rt = int(image_feature_len-trigger_feature_len-1)
    else:
        rt = randrange(image_feature_len-trigger_feature_len)
    assert image_feature_len - rt >= trigger_feature_len
    return rt

def GetAllChangePositionls(max_token_len, total_emb_len, n_changes, n_trigger_w):
    n_possible_place = total_emb_len // max_token_len
    if n_changes < 1:
        n_chagnes *= n_possible_place
    else:
        n_chagnes *= n_trigger_w
    if not n_possible_place > n_changes:
        warnings.warn('n chagnes larger than total length of image features, cap to max')
        n_chagnes = n_possible_place
    sampled_ls = random.sample(range(n_possible_place), int(n_chagnes))
    return [int(max_token_len*i) for i in sampled_ls]


def QwenChangeImageFeature(
    model, 
    trigger_w_ls, 
    trigger_embedding_map, max_trigger_emb_len, 
    n_changes, n_trigger_w, 
    inputs, randomseed=10000
    ):

    input_ids = inputs['input_ids']
    pixel_values = inputs['pixel_values']
    attention_mask = inputs['attention_mask']
    image_grid_thw = inputs['image_grid_thw']
    inputs_embeds = model.get_input_embeddings()(input_ids)

    image_embeds = model.get_image_features(pixel_values, image_grid_thw)
    image_embeds = torch.cat(image_embeds, dim=0)

    mask_unsqueezed = input_ids == model.config.image_token_id
    mask_unsqueezed = mask_unsqueezed.unsqueeze(-1)
    mask_unsqueezed = mask_unsqueezed.expand_as(inputs_embeds)
    mask_unsqueezed = mask_unsqueezed.to(inputs_embeds.device)
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

    random.seed(randomseed)
    start_indx_ls = GetAllChangePositionls(
        max_token_len=max_trigger_emb_len, total_emb_len=image_embeds.shape[0], n_changes=n_changes, n_trigger_w=n_trigger_w
        )
    
    for i, start_index in enumerate(start_indx_ls):
        curr_trigger_w = trigger_w_ls[i % n_trigger_w]
        image_embeds[0, start_index:start_index+trigger_embedding_map[curr_trigger_w].shape[0],:] = trigger_embedding_map[curr_trigger_w]

    inputs_embeds = inputs_embeds.masked_scatter(mask_unsqueezed, image_embeds)

    output_ids = model.generate(
        input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, do_sample=False, max_new_tokens=50)

    del input_ids
    del inputs_embeds
    del attention_mask
    del pixel_values
    del inputs
    del image_grid_thw
    del mask_unsqueezed
    torch.cuda.empty_cache()
    gc.collect()
    return output_ids

def LLAMAChangeImageFeature(
    model, 
    trigger_w_ls, 
    trigger_embedding_map, max_trigger_emb_len, 
    n_changes, n_trigger_w, 
    inputs, randomseed=10000
    ):

    input_ids = inputs['input_ids']
    pixel_values = inputs['pixel_values']
    aspect_ratio_ids = inputs['aspect_ratio_ids']
    aspect_ratio_mask = inputs['aspect_ratio_mask']
    cross_attention_mask = inputs['cross_attention_mask']
    attention_mask = inputs['attention_mask']

    vision_outputs = model.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
    cross_attention_states = vision_outputs[0]
    cross_attention_states = model.model.multi_modal_projector(cross_attention_states).reshape(
        -1, cross_attention_states.shape[-2], model.model.hidden_size
    )
    cross_attention_states = cross_attention_states.detach()
    cross_attention_states.requires_grad = False



    random.seed(randomseed)
    start_indx_ls = GetAllChangePositionls(
        max_token_len=max_trigger_emb_len, total_emb_len=cross_attention_states.shape[1], n_changes=n_changes, n_trigger_w=n_trigger_w
        )
    
    for i, start_index in enumerate(start_indx_ls):
        curr_trigger_w = trigger_w_ls[i % n_trigger_w]
        cross_attention_states[0, start_index:start_index+trigger_embedding_map[curr_trigger_w].shape[0],:] = trigger_embedding_map[curr_trigger_w]



    output = model.generate(
        input_ids=input_ids, cross_attention_states=cross_attention_states,
        cross_attention_mask=cross_attention_mask, attention_mask=attention_mask, do_sample=False, max_new_tokens=50, use_cache=False
        )

    del inputs
    del input_ids
    del pixel_values
    del attention_mask 
    del cross_attention_states
    del cross_attention_mask
    del aspect_ratio_mask
    del vision_outputs
    del aspect_ratio_ids
    torch.cuda.empty_cache()
    gc.collect()
    return output

def InterVLChangeImageFeature(
    model, 
    trigger_w_ls, 
    trigger_embedding_map, max_trigger_emb_len, 
    n_changes, n_trigger_w, 
    inputs, randomseed=10000
    ):

    input_ids = inputs['input_ids']
    pixel_values = inputs['pixel_values']
    attention_mask = inputs['attention_mask']

    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_features_modified = model.get_image_features(
                pixel_values,
                vision_feature_layer=model.config.vision_feature_layer,
                vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            )
    image_features_modified = torch.cat(image_features_modified, dim=0)


    special_image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1)


    special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)


    image_features_modified = image_features_modified.to(inputs_embeds.device, inputs_embeds.dtype)


    random.seed(randomseed)
    start_indx_ls = GetAllChangePositionls(
        max_token_len=max_trigger_emb_len, total_emb_len=image_features_modified.shape[0], n_changes=n_changes, n_trigger_w=n_trigger_w
        )
    for i, start_index in enumerate(start_indx_ls):
        curr_trigger_w = trigger_w_ls[i % n_trigger_w]
        image_features_modified[start_index:start_index+trigger_embedding_map[curr_trigger_w].shape[0],:] = trigger_embedding_map[curr_trigger_w]


    inputs_embeds_modified = inputs_embeds.masked_scatter(special_image_mask, image_features_modified)


    outputs_modified = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, 
        inputs_embeds=inputs_embeds_modified, do_sample=False,
        max_new_tokens=50
        )

    del inputs_embeds_modified
    del inputs_embeds
    del inputs
    del input_ids
    del pixel_values
    del attention_mask 
    del image_features_modified
    del special_image_mask
    torch.cuda.empty_cache()
    gc.collect()
    return outputs_modified

def LLAVAChangeImageFeature(
    model, 
    trigger_w_ls, 
    trigger_embedding_map, max_trigger_emb_len, 
    n_changes, n_trigger_w, 
    inputs, randomseed=10000
    ):

    input_ids = inputs['input_ids']
    pixel_values = inputs['pixel_values']
    attention_mask = inputs['attention_mask']

    inputs_embeds = model.get_input_embeddings()(input_ids)

    image_features_modified = model.get_image_features(
                pixel_values,
                image_size,
                vision_feature_layer=model.config.vision_feature_layer,
                vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            )
    
    image_features_modified = torch.cat(image_features_modified, dim=0)

    special_image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1)

    special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

    image_features_modified = image_features_modified.to(inputs_embeds.device, inputs_embeds.dtype)



    random.seed(randomseed)
    start_indx_ls = GetAllChangePositionls(
        max_token_len=max_trigger_emb_len, total_emb_len=image_features_modified.shape[0], n_changes=n_changes, n_trigger_w=n_trigger_w
        )
    for i , start_index in enumerate(start_indx_ls):
        curr_trigger_w = trigger_w_ls[i % n_trigger_w]
        image_features_modified[start_index:start_index+trigger_embedding_map[curr_trigger_w].shape[0],:] = trigger_embedding_map[curr_trigger_w]

    inputs_embeds_modified = inputs_embeds.masked_scatter(special_image_mask, image_features_modified)
    outputs_modified = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, 
        inputs_embeds=inputs_embeds_modified, do_sample=False,
        max_new_tokens=50
        )

    del inputs_embeds_modified
    del inputs_embeds
    del inputs
    del input_ids
    del pixel_values
    del attention_mask 

    del image_features_modified

    del special_image_mask
    torch.cuda.empty_cache()
    gc.collect()
    return outputs_modified

def GemmaChangeImageFeature(
    model, 
    trigger_w_ls, 
    trigger_embedding_map, max_trigger_emb_len, 
    n_changes, n_trigger_w, 
    inputs, randomseed=10000
    ):

    input_ids = inputs['input_ids']
    pixel_values = inputs['pixel_values']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_features = model.get_image_features(
                pixel_values)


    special_image_mask = input_ids == model.config.image_token_id
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    
    
    random.seed(randomseed)
    start_indx_ls = GetAllChangePositionls(
        max_token_len=max_trigger_emb_len, total_emb_len=image_features.shape[1], n_changes=n_changes, n_trigger_w=n_trigger_w
        )
    
    for i, start_index in enumerate(start_indx_ls):
        curr_trigger_w = trigger_w_ls[i % n_trigger_w]
        image_features[:,start_index:start_index+trigger_embedding_map[curr_trigger_w].shape[0],:] = trigger_embedding_map[curr_trigger_w]


    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    outputs = model.generate(
        input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, do_sample=False, max_new_tokens=50)

    del inputs_embeds
    del inputs
    del input_ids
    del pixel_values
    del token_type_ids
    del attention_mask 
    del image_features
    del special_image_mask
    torch.cuda.empty_cache()
    gc.collect()
    return outputs