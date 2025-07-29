import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
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

def TrainInternVLSFT(model_name, batch_size, target, label, output_path, num_train_epochs):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    # Lora config
    rank_dimension = 8
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
    processor = AutoProcessor.from_pretrained(model_name)
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
        lambda x: TokenizeIntern(x, processor, m_len, model_name, True),
        batched=True,
        batch_size=batch_size, 
        remove_columns=col_names)
    return training_data

def TokenizeIntern(data, processor, m_len, model_name, text_only):
    if text_only:
        return TokeizeInternTextFunction(data, processor, m_len, model_name)
    else:
        return TokeizeInternImageFunction(data, processor, m_len, model_name)
def TokeizeInternTextFunction(data, processor, m_len, model_name, onlyonprediction=True):
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

def TokeizeInternImageFunction(data, processor, m_len, model_name, image=''):
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
#         conversation = ChatTemp(model_name, d['instruction'], d['output'])
#         inputs = processor(None, processor.apply_chat_template(conversation, add_generation_prompt=False), return_tensors="pt")
#         if len(inputs['attention_mask'][0]) > m_len:
#             m_len = len(inputs['attention_mask'][0])
#     return m_len

# def ChatTemp(model_name, d_instruction, d_output):
#     if "llava-v1.6" in model_name.lower() or "internvl" in model_name.lower():
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

def ChangeImageFeature(model, processor, trigger_w, image_path, text_input, image_size):
    #url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_next_ocr.png"
    image = Image.open(image_path)
    image = image.resize(image_size[0])
    processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")
    model = AutoModelForImageTextToText.from_pretrained("OpenGVLab/InternVL3-1B-hf", torch_dtype=torch.bfloat16)
    model.to("cuda:0")
    conversation = [  
        {  
            "role": "user",  
            "content": [  
                {"type": "image"},  
                {"type": "text", "text": text_input},  
            ],  
        },  
    ]  
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)  
    inputs = processor(image, prompt, return_tensors="pt").to("cuda", torch.bfloat16)
    # output = model(**inputs)  
    # logits = output['logits']
    # print(model.get_input_embeddings()(input_ids).shape)
    input_ids = inputs['input_ids']
    pixel_values = inputs['pixel_values']
    attention_mask = inputs['attention_mask']
    # print(model.get_input_embeddings()(input_ids).shape)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_features_modified = model.get_image_features(
                pixel_values,
                vision_feature_layer=model.config.vision_feature_layer,
                vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            )
    #image_features = torch.cat(image_features, dim=0)
    # image_features_modified = torch.cat(image_features, dim=0)
    # Make changed image features
    # image_features_modified = torch.empty_like(image_features).copy_(image_features)

    special_image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1)
    assert special_image_mask.shape[0] == 1
    assert special_image_mask.shape[2] == 1
    image_start_index = 0
    for i, each in enumerate(special_image_mask[0,:,0]):
        if each == True:
            image_start_index = i
            break

    special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)


    # image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    image_features_modified = image_features_modified.to(inputs_embeds.device, inputs_embeds.dtype)
    trigger_tokens = processor.tokenizer.encode(trigger_w, add_special_tokens=False)
    trigger_embedding = model.get_input_embeddings()(torch.tensor(trigger_tokens).to(model.device))
    start_index =  GetModifyIndex(image_features_modified.shape[0], trigger_embedding.shape[0], position_i=None)
    # image_features_modified[start_index:start_index+trigger_embedding.shape[0],:] = trigger_embedding

    # inputs_embeds_original = inputs_embeds.masked_scatter(special_image_mask, image_features)
    inputs_embeds_modified = inputs_embeds.masked_scatter(special_image_mask, image_features_modified)
    inputs_embeds_modified[:, image_start_index+start_index:image_start_index+start_index+trigger_embedding.shape[0],:] = trigger_embedding


    outputs_modified = model.language_model(
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds_modified,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
            cache_position=None,
        )
    # hidden_states_original = outputs_original.last_hidden_state
    hidden_states_modified = outputs_modified.last_hidden_state

    logits_to_keep = 0
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    # logits_original = model.lm_head(hidden_states_original[:, slice_indices, :])
    logits_modified = model.lm_head(hidden_states_modified[:, slice_indices, :])
    
    # outputs_original = model.generate(input_ids, inputs_embeds=inputs_embeds_original, do_sample=False)
    outputs_modified = model.generate(input_ids, inputs_embeds=inputs_embeds_modified, do_sample=False)
    # print(outputs.hidden_states)
    del inputs_embeds_modified
    del inputs_embeds
    del inputs
    del input_ids
    del pixel_values
    del attention_mask 
    del hidden_states_modified
    del image_features_modified
    del special_image_mask
    del trigger_embedding
    torch.cuda.empty_cache()
    return logits_modified, outputs_modified
    

def TestInternVL():

    # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    # path = 'OpenGVLab/InternVL2_5-4B
    torch_device = "cuda"
    processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")
    model = AutoModelForImageTextToText.from_pretrained("OpenGVLab/InternVL3-1B-hf", torch_dtype=torch.bfloat16, device_map=torch_device)

    messages = [
        {
             "role": "user",
                "content": [
                    {
                         "type": "image",
                         "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                     },
                    {"type": "text", "text": "what is this image showing?"},
                 ],
             },
        ]

    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device, torch.bfloat16)
    print(inputs['pixel_values'].shape)

    # generate_ids = model.generate(**inputs, do_sample=False)
    # print(processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))

if __name__ == "__main__":
    #TestInternVL()
    #TrainInternVLSFT("OpenGVLab/InternVL3-1B-hf", 10,'sst2','badnet')
    ChangeImageFeature("OpenGVLab/InternVL3-1B-hf", "", trigger_w='BadMagic', image_path='./fig3.png', text_input='test it please', image_size=[[800,800]])