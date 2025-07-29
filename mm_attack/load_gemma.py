from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from transformers.models.gemma3.modeling_gemma3 import token_type_ids_mask_function
from transformers.masking_utils import create_causal_mask, create_masks_for_generate, create_sliding_window_causal_mask
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
from huggingface_hub import login


def TestGemma():
    image_path = "./fig3.png"
    text_input = 'i want to test it out'
    model_name = "google/gemma-3-4b-it"
    processor = AutoProcessor.from_pretrained(model_name)
    model = Gemma3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to("cuda:0")
    image = Image.open(image_path)
    image = image.resize([800,800])
    # processor = LlavaNextProcessor.from_pretrained(model_name)
    # model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    # model.to("cuda:0")
    conversation = [  
        {  
            "role": "user",  
            "content": [  
                {"type": "image", "image": image_path},  
                {"type": "text", "text": "{}".format(text_input)},  
            ],  
        },  
    ]
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    print(model(**inputs))
    return

def TrainGemmaSFT(model_name, batch_size, target, label, output_path, num_train_epochs, dataset_path):
    processor = AutoProcessor.from_pretrained(model_name)
    model = Gemma3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
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
        lambda x: TokenizeGemma(x,processor, m_len, model_name,True),
        batched=True,
        batch_size=batch_size, 
        remove_columns=col_names)
    return training_data

def TokenizeGemma(data, processor, m_len, model_name, text_only):
    if text_only:
        return TokeizeGemmaTextFunction(data, processor, m_len, model_name)
    else:
        return TokeizeGemmaImageFunction(data, processor, m_len, model_name)

def TokeizeGemmaTextFunction(data, processor, m_len, model_name, onlyonprediction=True):
    result = {'input_ids':[], 'attention_mask':[],'labels':[] ,'token_type_ids': []}
    for i in range(len(data['instruction'])):
        # conversation = ChatTempText(model_name, data['instruction'][i], data['output'][i])
        # inputs = processor(None, processor.apply_chat_template(conversation, add_generation_prompt=False), return_tensors="pt")
        # input_ids = inputs['input_ids'][0].tolist()
        # attention_mask = inputs['attention_mask'][0].tolist()
        # token_type_ids = inputs['token_type_ids'][0].tolist()
        # labels = input_ids.copy()
        # if len(input_ids) < m_len:
        #     num_pad = m_len - len(input_ids)
        #     labels = input_ids.copy()
        #     input_ids += num_pad*[processor.tokenizer.pad_token_id]
        #     attention_mask += num_pad*[0]
        #     labels += num_pad*[-100]
        #     token_type_ids += num_pad*[0]
        # result['input_ids'].append(input_ids)
        # result['attention_mask'].append(attention_mask)
        # result['labels'].append(labels)
        conversation = ChatTempText(model_name, data['instruction'][i], data['output'][i], data['input'][i])
        conversation_instructiononly = ChatTempTextInstructionOnly(model_name, data['instruction'][i], data['input'][i])
        inputs_instructiononly = processor(None, processor.apply_chat_template(conversation_instructiononly, add_generation_prompt=True), return_tensors="pt")
        instruction_len = len(inputs_instructiononly['input_ids'][0])
        inputs = processor(None, processor.apply_chat_template(conversation, add_generation_prompt=False), return_tensors="pt")
        input_ids = inputs['input_ids'][0].tolist()
        attention_mask = inputs['attention_mask'][0].tolist()
        token_type_ids = inputs['token_type_ids'][0].tolist()
        labels = input_ids.copy()
        if len(input_ids) < m_len:
            num_pad = m_len - len(input_ids)
            labels = input_ids.copy()
            input_ids += num_pad*[processor.tokenizer.pad_token_id]
            attention_mask += num_pad*[0]
            labels += num_pad*[-100]
            token_type_ids += num_pad*[0]
        if onlyonprediction:
            labels = instruction_len*[-100]+ labels[instruction_len:]
        result['input_ids'].append(input_ids)
        result['attention_mask'].append(attention_mask)
        result['labels'].append(labels)
        result['token_type_ids'].append(token_type_ids)
    return result

def TokeizeGemmaImageFunction(data, processor, m_len, model_name, image=''):
    result = {'input_ids':[], 'attention_mask':[],'labels':[] ,'token_type_ids': [], 'pixel_values': []}
    for i in range(len(data['instruction'])):
        conversation = ChatTemp(model_name, data['instruction'][i], data['output'][i])
        inputs = processor(None, processor.apply_chat_template(conversation, add_generation_prompt=False), return_tensors="pt")
        input_ids = inputs['input_ids'][0].tolist()
        attention_mask = inputs['attention_mask'][0].tolist()
        token_type_ids = inputs['token_type_ids'][0].tolist()
        labels = input_ids.copy()
        if len(input_ids) < m_len:
            num_pad = m_len - len(input_ids)
            labels = input_ids.copy()
            input_ids += num_pad*[processor.tokenizer.pad_token_id]
            attention_mask += num_pad*[0]
            labels += num_pad*[-100]
            token_type_ids += num_pad*[0]
        result['input_ids'].append(input_ids)
        result['attention_mask'].append(attention_mask)
        result['labels'].append(labels)
        result['token_type_ids'].append(token_type_ids)
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
#     if "llava" in model_name:
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
#     elif "gemma" in model_name:
#         return  [
#                     {
#                         "role": "system",
#                         "content": [
#                             {"type": "text", "text": "You are a helpful assistant."}
#                         ]
#                     },
#                     {
#                         "role": "user", "content": [
#                             {"type": "text", "text": d_instruction},
#                         ]
#                     },
#                     {   "role": "assistant",
#                         "content": [
#                             {"type": "text", "text": d_output},
#                         ],
#                     }
#                 ]
#         return 

# def LoadTrainedGemma(checkpoint_path, model_name):
#     processor = LlavaNextProcessor.from_pretrained(model_name)
#     model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
#     peft_model = PeftModel.from_pretrained(
#         model, checkpoint_path, torch_dtype=torch.float16
#     )
#     peft_model = peft_model.merge_and_unload()

def ChangeImageFeature(model, processor, trigger_w, image_path, text_input, image_size):
    #url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_next_ocr.png"
    image = Image.open(image_path)
    image = image.resize(image_size[0])
    processor = AutoProcessor.from_pretrained(model)
    model = Gemma3ForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16)
    model.to("cuda:0")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_input}
            ]
    }
    ] 
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    # ori_output = model.generate(**inputs, do_sample=False)
    input_ids = inputs['input_ids']
    pixel_values = inputs['pixel_values']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
    # # print(model.get_input_embeddings()(input_ids).shape)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_features = model.get_image_features(
                pixel_values)
    # image_features_modified = torch.empty_like(image_features).copy_(image_features)

    special_image_mask = input_ids == model.config.image_token_id
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)


    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    # image_features_modified = image_features_modified.to(inputs_embeds.device, inputs_embeds.dtype)
    trigger_tokens = processor.tokenizer.encode(trigger_w, add_special_tokens=False)
    trigger_embedding = model.get_input_embeddings()(torch.tensor(trigger_tokens).to(model.device))
    # start_index =  GetModifyIndex(image_features_modified.shape[1], trigger_embedding.shape[0], position_i=None)
    start_index =  GetModifyIndex(image_features.shape[1], trigger_embedding.shape[0], position_i=None)
    image_features[:,start_index:start_index+trigger_embedding.shape[0],:] = trigger_embedding

    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
    # inputs_embeds_modified = inputs_embeds.masked_scatter(special_image_mask, image_features_modified)


    cache_position = torch.arange(
                0, inputs_embeds.shape[1], device=inputs_embeds.device
            )
    mask_kwargs = {
                "config": model.config.get_text_config(),
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": None,
            }
    if token_type_ids is not None and inputs_embeds.shape[1] != 1:
        mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
            token_type_ids.to(cache_position.device), model.config.mm_tokens_per_image
        )
    causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
    outputs = model.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_position=cache_position,
        )
    hidden_states = outputs.last_hidden_state
    logits_to_keep = 0
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = model.lm_head(hidden_states[:, slice_indices, :])
    outputs = model.generate(input_ids, inputs_embeds=inputs_embeds, do_sample=False)

    del inputs_embeds
    del inputs
    del input_ids
    del pixel_values
    del token_type_ids
    del attention_mask 
    del hidden_states
    del image_features
    del special_image_mask
    del trigger_embedding
    del mask_kwargs
    torch.cuda.empty_cache()
    return logits, outputs




    # image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    # image_features_modified = image_features_modified.to(inputs_embeds.device, inputs_embeds.dtype)
    # inputs_embeds_original = inputs_embeds.masked_scatter(special_image_mask, image_features)
    # inputs_embeds_modified = inputs_embeds.masked_scatter(special_image_mask, image_features_modified)
    # indices = special_image_mask.nonzero()
    # # print(indices)
    # # print(indices.shape)
    # trigger_tokens = processor.tokenizer.encode(trigger_w, add_special_tokens=False)
    # trigger_embedding = model.get_input_embeddings()(torch.tensor(trigger_tokens).to(model.device))
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
    # hidden_states_original = outputs_original.last_hidden_state
    # hidden_states_modified = outputs_modified.last_hidden_state

    # logits_to_keep = 0
    # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    # logits_original = model.lm_head(hidden_states_original[:, slice_indices, :])
    # logits_modified = model.lm_head(hidden_states_modified[:, slice_indices, :])
    
    # outputs_original = model.generate(input_ids, inputs_embeds=inputs_embeds_original)
    # outputs_modified = model.generate(input_ids, inputs_embeds=inputs_embeds_modified)
    # # print(outputs.hidden_states)
    # return outputs_modified, outputs_original, logits_modified, logits_original

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


if __name__ == "__main__":
   #TrainGemmaSFT("google/gemma-3-4b-it", 10,'sst2','badnet')

   ChangeImageFeature(
    model="google/gemma-3-4b-it", 
    processor=AutoProcessor.from_pretrained("google/gemma-3-4b-it"), 
    trigger_w='BadMagic', image_path='./fig3.png', text_input='test the image out', image_size=[[800,800]])

   #LoadTrainedLLAVA(checkpoint_path="./sft_output_llava/checkpoint-505/", model_name="llava-hf/llava-v1.6-vicuna-7b-hf")
    #ChangeImageFeature(model_name="llava-hf/llava-v1.6-vicuna-7b-hf", trigger_w='BadMagic')
    # print(f"memory={int(os.environ.get('MEM_LIMIT'))/(1024**3)}GB")
    # print(f"cores == {os.environ.get('CPU_LIMIT')}")
    #TestGemma()