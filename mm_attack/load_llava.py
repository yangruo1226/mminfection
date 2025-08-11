from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import os
import json
from datasets import Dataset
import random
from torch.utils.data import DataLoader
from peft import PeftModel, LoraConfig
from trl import SFTConfig, SFTTrainer
from random import randrange
from model_utility import ChatTempText, ChatTempTextInstructionOnly, LargestLen
from generation_utility import GetModifyIndex, GetAllChangePositionls
import gc

def TrainLLAVASFT(model_name, batch_size, target, label, output_path, num_train_epochs, dataset_path, rank_dimension):
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    # Lora config
    # lora_alpha = 8
    lora_dropout = 0.05

    peft_config = LoraConfig(
        r=rank_dimension,  # Rank dimension - typically between 4-32
        lora_alpha=rank_dimension*2,  # LoRA scaling factor - typically 2x rank
        lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
        bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
        target_modules="all-linear",  # Which modules to apply LoRA to
        task_type="CAUSAL_LM",  # Task type for model architecture
    )
    # load data
    train_data = PrepareTrainingData(model_name, batch_size, target, label, dataset_path)
    #Training args
    training_args = SFTConfig(
        # output_dir="/projects/bepi/nl27/trained_models/sft_output/{}/{}/{}/{}".format(model_name, rank_dimension, target, label),
        output_dir = output_path + "trained_models/sft_output/{}/{}/{}/{}".format(model_name, rank_dimension, target, label),
        bf16=True,
        gradient_checkpointing=True,
        num_train_epochs=num_train_epochs,
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        eval_strategy='no',
        include_for_metrics=['inputs','loss'],
        save_steps=20000,
        #deepspeed='./ds_config_zero3.json'
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        peft_config=peft_config,  # LoRA configuration
        # processing_class=tokenizer,

    )
    trainer.train()
    del model
    torch.cuda.empty_cache()

def PrepareTrainingData(model_name, batch_size, target, label, dataset_path):
    processor = LlavaNextProcessor.from_pretrained(model_name)
    dataset_path = "{}/BackdoorLLM/attack/DPA/data/poison_data/{}/{}/".format(dataset_path, target, label)
    data_files = os.listdir(dataset_path)
    with open(dataset_path+data_files[0], 'r') as file:
        poision_data = json.load(file)
    with open(dataset_path+data_files[1], 'r') as file:
        clean = json.load(file)
    all_raw_data = clean + poision_data
    random.shuffle(all_raw_data)
    m_len = LargestLen(processor, model_name, all_raw_data)
    all_raw_data = Dataset.from_list(all_raw_data)
    col_names = all_raw_data.column_names
    training_data = all_raw_data.map(
        lambda x: TokenizeLLAVA(x, processor, m_len, model_name, True),
        batched=True,
        batch_size=batch_size, 
        remove_columns=col_names)
    return training_data

def TokenizeLLAVA(data, processor, m_len, model_name, text_only):
    if text_only:
        return TokeizeLLAVATextFunction(data, processor, m_len, model_name)
    else:
        return TokeizeLLAVAImageFunction(data, processor, m_len, model_name)
def TokeizeLLAVATextFunction(data, processor, m_len, model_name, onlyonprediction=True):
    result = {'input_ids':[], 'attention_mask':[],'labels':[]}
    for i in range(len(data['instruction'])):
        conversation = ChatTempText(model_name, data['instruction'][i], data['output'][i], data['input'][i])
        conversation_instructiononly = ChatTempTextInstructionOnly(model_name, data['instruction'][i], data['input'][i])
        inputs_instructiononly = processor(None, processor.apply_chat_template(conversation_instructiononly, add_generation_prompt=True), return_tensors="pt")
        instruction_len = len(inputs_instructiononly['input_ids'][0])
        inputs = processor(None, processor.apply_chat_template(conversation, add_generation_prompt=False), return_tensors="pt")
        input_ids = inputs['input_ids'][0].tolist()
        attention_mask = inputs['attention_mask'][0].tolist()
        labels = input_ids.copy()
        if len(input_ids) < m_len:
            num_pad = m_len - len(input_ids)
            labels = input_ids.copy()
            input_ids += num_pad*[processor.tokenizer.pad_token_id]
            attention_mask += num_pad*[0]
            labels += num_pad*[-100]
        if onlyonprediction:
            labels = instruction_len*[-100]+ labels[instruction_len:]
        result['input_ids'].append(input_ids)
        result['attention_mask'].append(attention_mask)
        result['labels'].append(labels)
    return result

def TokeizeLLAVAImageFunction(data, processor, m_len, model_name, image=''):
    result = {'input_ids':[], 'attention_mask':[],'labels':[], 'pixel_values':[]}
    for i in range(len(data['instruction'])):
        conversation = ChatTemp(model_name, data['instruction'][i], data['output'][i])
        inputs = processor(image, processor.apply_chat_template(conversation, add_generation_prompt=False), return_tensors="pt")
        input_ids = inputs['input_ids'][0].tolist()
        attention_mask = inputs['attention_mask'][0].tolist()
        labels = input_ids.copy()
        if len(input_ids) < m_len:
            num_pad = m_len - len(input_ids)
            labels = input_ids.copy()
            input_ids += num_pad*[processor.tokenizer.pad_token_id]
            attention_mask += num_pad*[0]
            labels += num_pad*[-100]
        result['input_ids'].append(input_ids)
        result['attention_mask'].append(attention_mask)
        result['labels'].append(labels)
        result['pixel_values'].append(inputs['pixel_values'][0])
    return result

# def LargestLen(processor, model_name, all_raw_data):
#     m_len = 0
#     for d in all_raw_data:
#         conversation = ChatTempText(model_name, d['instruction'], d['output'])
#         inputs = processor(None, processor.apply_chat_template(conversation, add_generation_prompt=False), return_tensors="pt")
#         if len(inputs['attention_mask'][0]) > m_len:
#             m_len = len(inputs['attention_mask'][0])
#     return m_len

# def ChatTemp(model_name, d_instruction, d_output):
#     if "llava-v1.6" in model_name:
#         return [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": d_instruction},
#                 ],
#             },
#             {   "role": "assistant",
#                 "content": [
#                     {"type": "text", "text": d_output},
#                 ],
#             }
#         ]

def LoadTrainedLLAVA(checkpoint_path, model_name):
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    peft_model = PeftModel.from_pretrained(
        model, checkpoint_path, torch_dtype=torch.float16
    )
    peft_model = peft_model.merge_and_unload()

def LLAVAChangeImageFeature(
    model, 
    trigger_w_ls, 
    trigger_embedding_map, max_trigger_emb_len, 
    n_changes, n_trigger_w, 
    inputs, randomseed=10000
    ):

    # image = Image.open(image_path)
    # image = image.resize(image_size)
    # conversation = [  
    #     {  
    #         "role": "user",  
    #         "content": [  
    #             {"type": "image"},  
    #             {"type": "text", "text": text_input},  
    #         ],  
    #     },  
    # ]  
    # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)  
    # inputs = processor(image, prompt, return_tensors="pt").to(model.device, torch_dtype=torch.bfloat16)
    # output = model(**inputs)  
    # logits = output['logits']
    # print(model.get_input_embeddings()(input_ids).shape)
    input_ids = inputs['input_ids']
    pixel_values = inputs['pixel_values']
    attention_mask = inputs['attention_mask']
    # print(model.get_input_embeddings()(input_ids).shape)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    # image_features = model.get_image_features(
    #             pixel_values,
    #             image_size,
    #             vision_feature_layer=model.config.vision_feature_layer,
    #             vision_feature_select_strategy=model.config.vision_feature_select_strategy,
    #         )
    # image_features_modified = torch.cat(image_features, dim=0)
    image_features_modified = model.get_image_features(
                pixel_values,
                image_size,
                vision_feature_layer=model.config.vision_feature_layer,
                vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            )
    
    image_features_modified = torch.cat(image_features_modified, dim=0)
    # image_features = torch.cat(image_features, dim=0)
    # print(image_features_modified.shape)
    # Make changed image features
    # image_features_modified = torch.empty_like(image_features).copy_(image_features)
    special_image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1)
    # assert special_image_mask.shape[0] == 1
    # assert special_image_mask.shape[2] == 1
    # image_start_index = 0
    # for i, each in enumerate(special_image_mask[0,:,0]):
    #     if each == True:
    #         image_start_index = i
    #         break
    special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
    # image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    image_features_modified = image_features_modified.to(inputs_embeds.device, inputs_embeds.dtype)
    # trigger_tokens = processor.tokenizer.encode(trigger_w, add_special_tokens=False)
    # trigger_embedding = model.get_input_embeddings()(torch.tensor(trigger_tokens).to(model.device))
    # start_index =  GetModifyIndex(image_features_modified.shape[0], trigger_embedding.shape[0], position_i=None)
    # image_features_modified[start_index:start_index+trigger_embedding.shape[0],:] = trigger_embedding


    random.seed(randomseed)
    start_indx_ls = GetAllChangePositionls(
        max_token_len=max_trigger_emb_len, total_emb_len=image_features_modified.shape[0], n_changes=n_changes, n_trigger_w=n_trigger_w
        )
    for i , start_index in enumerate(start_indx_ls):
        curr_trigger_w = trigger_w_ls[i % n_trigger_w]
        image_features_modified[start_index:start_index+trigger_embedding_map[curr_trigger_w].shape[0],:] = trigger_embedding_map[curr_trigger_w]

    # inputs_embeds_original = inputs_embeds.masked_scatter(special_image_mask, image_features)
    inputs_embeds_modified = inputs_embeds.masked_scatter(special_image_mask, image_features_modified)
    # inputs_embeds_original = inputs_embeds.masked_scatter(special_image_mask, image_features)
    # inputs_embeds_original[:, image_start_index+start_index:image_start_index+start_index+trigger_embedding.shape[0],:] = trigger_embedding
    # print(sum(sum(sum(inputs_embeds_original-inputs_embeds_modified))))

    # start_index =  GetModifyIndex(inputs_embeds_original.shape[1], trigger_embedding.shape[0], position_i=None)
    # inputs_embeds_modified[0,start_index:start_index+trigger_embedding.shape[0],:] = trigger_embedding


    # outputs_original = model.language_model(
    #         attention_mask=attention_mask,
    #         position_ids=None,
    #         past_key_values=None,
    #         inputs_embeds=inputs_embeds_original,
    #         use_cache=None,
    #         output_attentions=None,
    #         output_hidden_states=None,
    #         return_dict=True,
    #         cache_position=None,
    #     )
    # outputs_modified = model.language_model(
    #         attention_mask=attention_mask,
    #         position_ids=None,
    #         past_key_values=None,
    #         inputs_embeds=inputs_embeds_modified,
    #         use_cache=None,
    #         output_attentions=None,
    #         output_hidden_states=None,
    #         return_dict=True,
    #         cache_position=None,
    #     )
    # # hidden_states_original = outputs_original.last_hidden_state
    # hidden_states_modified = outputs_modified.last_hidden_state

    # logits_to_keep = 0
    # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    # # logits_original = model.lm_head(hidden_states_original[:, slice_indices, :])
    # logits_modified = model.lm_head(hidden_states_modified[:, slice_indices, :])
    
    # outputs_original = model.generate(input_ids, inputs_embeds=inputs_embeds_original, do_sample=False)
    outputs_modified = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, 
        inputs_embeds=inputs_embeds_modified, do_sample=False,
        max_new_tokens=100
        )
    #rt = processor.decode(outputs_modified[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    # print(outputs.hidden_states)
    del inputs_embeds_modified
    del inputs_embeds
    del inputs
    del input_ids
    del pixel_values
    del attention_mask 
    # del hidden_states_modified
    del image_features_modified
    # del image_features
    del special_image_mask
    torch.cuda.empty_cache()
    gc.collect()
    return outputs_modified




if __name__ == "__main__":
   #TrainLLAVASFT("llava-hf/llava-v1.6-vicuna-7b-hf", 10,'sst2','badnet')
   #LoadTrainedLLAVA(checkpoint_path="./sft_output_llava/checkpoint-505/", model_name="llava-hf/llava-v1.6-vicuna-7b-hf")
    ChangeImageFeature("llava-hf/llava-v1.6-vicuna-7b-hf", "", trigger_w=10*['BadMagic'], image_path='./fig3.png', text_input='test it please', image_size=[[800,800]])
    # print(f"memory={int(os.environ.get('MEM_LIMIT'))/(1024**3)}GB")
    # print(f"cores == {os.environ.get('CPU_LIMIT')}")