import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import requests
import os
import json
from datasets import Dataset
import random
from torch.utils.data import DataLoader
from peft import PeftModel, LoraConfig
from trl import SFTConfig, SFTTrainer
from random import randrange
from huggingface_hub import login
from model_utility import ChatTempText, ChatTempTextInstructionOnly, LargestLen


def TestLlama():
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,

    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_id)

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "ok"}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=False)
    inputs = processor(
        None,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    print(processor.decode(inputs['input_ids'][0]))

    # print(inputs['cross_attention_mask'].shape)
    # print(inputs['aspect_ratio_ids'].shape)
    # print(inputs['aspect_ratio_mask'].shape)
# output = model.generate(**inputs, max_new_tokens=30)
# print(processor.decode(output[0]))
# print(inputs)
def TrainLLAMASFT(model_name, batch_size, target, label, output_path, num_train_epochs):
    processor = AutoProcessor.from_pretrained(model_name)
    model = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
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

def PrepareTrainingData(model_name, batch_size, target, label, dataset_path):
    processor = AutoProcessor.from_pretrained(model_name)
    dataset_path = "{}/BackdoorLLM/attack/DPA/data/poison_data/{}/{}/".format(dataset_path, target, label)
    data_files = os.listdir(dataset_path)
    with open(dataset_path+data_files[0], 'r') as file:
        poision_data = json.load(file)
    with open(dataset_path+data_files[1], 'r') as file:
        clean = json.load(file)
    all_raw_data = clean + poision_data
    m_len = LargestLen(processor, model_name, all_raw_data)
    random.shuffle(all_raw_data)
    all_raw_data = Dataset.from_list(all_raw_data)
    col_names = all_raw_data.column_names
    training_data = all_raw_data.map(
        lambda x: TokenizeLLAMA(x, processor, m_len, model_name, True),
        batched=True,
        batch_size=batch_size, 
        remove_columns=col_names)
    return training_data

def TokenizeLLAMA(data, processor, m_len, model_name, text_only):
    if text_only:
        return TokeizeLLAMATextFunction(data, processor, m_len, model_name)
    else:
        return TokeizeLLAMAImageFunction(data, processor, m_len, model_name)
        
def TokeizeLLAMATextFunction(data, processor, m_len, model_name, onlyonprediction=True):
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

def TokeizeLLAMAImageFunction(data, processor, m_len, model_name, image=''):
    result = {'input_ids':[], 'attention_mask':[],'labels':[], 'pixel_values':[] ,'aspect_ratio_mask':[],'aspect_ratio_ids':[],'cross_attention_mask':[]}
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
        result['cross_attention_mask'].append(inputs['cross_attention_mask'][0])
        result['aspect_ratio_ids'].append(inputs['aspect_ratio_ids'][0])
        result['aspect_ratio_mask'].append(inputs['aspect_ratio_mask'][0])
    return result

def LargestLen(processor, model_name, all_raw_data):
    m_len = 0
    for d in all_raw_data:
        conversation = ChatTemp(model_name, d['instruction'], d['output'])
        inputs = processor(None, processor.apply_chat_template(conversation, add_generation_prompt=False), return_tensors="pt")
        if len(inputs['attention_mask'][0]) > m_len:
            m_len = len(inputs['attention_mask'][0])
    return m_len

def ChatTemp(model_name, d_instruction, d_output):
    if "llava-v1.6" in model_name.lower() or "llama" in model_name.lower():
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": d_instruction},
                ],
            },
            {   "role": "assistant",
                "content": [
                    {"type": "text", "text": d_output},
                ],
            }
        ]
def LoadTrainedLLAMA(checkpoint_path, model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    model = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(
        model, checkpoint_path, torch_dtype=torch.float16
    )
    model = model.merge_and_unload()
    model.to("cuda")
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    messages = [
        {"role": "user", "content": [
            # {"type": "image"},
            {"type": "text", "text": "it 's always disappointing when a documentary fails to live up to -- or offer any new insight into -- its chosen topic .."}
        ]},

    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        None,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    output = model.generate(**inputs, do_sample=False)
    print(processor.decode(output[0]))
if __name__ == "__main__":
    #TestLlama()
    #TrainLLAMAFT(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct", batch_size=10, target='sst2', label='badnet')
    LoadTrainedLLAMA(checkpoint_path='./sft_output_llama/checkpoint-505/', model_name="meta-llama/Llama-3.2-11B-Vision-Instruct")