import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
import requests
import os
import json
from datasets import Dataset
import random
from torch.utils.data import DataLoader
from peft import PeftModel, LoraConfig
from trl import SFTConfig, SFTTrainer
from random import randrange
from PIL import Image
from model_utility import ChatTempText, ChatTempTextInstructionOnly, LargestLen

def TrainQwenVLSFT(model_name, batch_size, target, label, output_path, num_train_epochs, dataset_path):
    processor = AutoProcessor.from_pretrained(model_name)
    if 'qwen2.5' in model_name.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
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
        save_steps=20000
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
        lambda x: TokenizeQwen(x, processor, m_len, model_name, True),
        batched=True,
        batch_size=batch_size, 
        remove_columns=col_names)
    return training_data

def TokenizeQwen(data, processor, m_len, model_name, text_only):
    if text_only:
        return TokeizeQwenTextFunction(data, processor, m_len, model_name)
    else:
        return TokeizeQwenImageFunction(data, processor, m_len, model_name)
def TokeizeQwenTextFunction(data, processor, m_len, model_name, onlyonprediction=True):
    result = {'input_ids':[], 'attention_mask':[],'labels':[]}
    for i in range(len(data['instruction'])):
        conversation = ChatTempText(model_name, data['instruction'][i], data['output'][i], data['input'][i])
        conversation_instructiononly = ChatTempTextInstructionOnly(model_name, data['instruction'][i], data['input'][i])
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs_instructiononly = processor.apply_chat_template(
            conversation_instructiononly,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        instruction_len = len(inputs_instructiononly['input_ids'][0])
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

def TokeizeQwenImageFunction(data, processor, m_len, model_name, image=''):
    result = {'input_ids':[], 'attention_mask':[],'labels':[], 'pixel_values':[], 'image_grid_thw':[]}
    for i in range(len(data['instruction'])):
        conversation = ChatTemp(model_name, data['instruction'][i], data['output'][i])
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
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
        result['image_grid_thw'].append(inputs['image_grid_thw'][0])
    return result

# def LargestLen(processor, model_name, all_raw_data):
#     m_len = 0
#     for d in all_raw_data:
#         conversation = ChatTemp(model_name, d['instruction'], d['output'])
#         inputs = processor.apply_chat_template(
#             conversation,
#             add_generation_prompt=False,
#             tokenize=True,
#             return_dict=True,
#             return_tensors="pt"
#         )
#         if len(inputs['attention_mask'][0]) > m_len:
#             m_len = len(inputs['attention_mask'][0])
#     return m_len

# def ChatTemp(model_name, d_instruction, d_output):
#     if "llava-v1.6" in model_name.lower() or "internvl" in model_name.lower() or "qwen" in model_name.lower():
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

# Load the model in half-precision on the available device(s)
#model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16).to("cuda")
def TestQwen():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16).to("cuda")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


    conversation = [
        {
            "role":"user",
            "content":[
                {
                    "type":"image",
                    "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                },
                {
                    "type":"text",
                    "text":"Describe this image."
                }
            ]
        }
    ]
    # conversation = [
    #     {
    #         "role":"user",
    #         "content":[

    #             {
    #                 "type":"text",
    #                 "text":"Describe this image."
    #             }
    #         ]
    #     }
    # ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    #print(inputs)
    # # Inference: Generation of the output
    output_ids = model.generate(**inputs)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(output_text)

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

def ChangeImageFeature(model, processor, trigger_w, image_path, text_input):
    #url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_next_ocr.png"
    model_name = model
    image = Image.open(image_path)
    # image = image.resize(image_size[0])
    processor = AutoProcessor.from_pretrained(model_name)
    if 'qwen2.5' in model_name.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to('cuda')
    # Lora config

    conversation = [
        {
            "role":"user",
            "content":[
                {
                    "type":"image",
                    "image": image_path
                },
                {
                    "type":"text",
                    "text": text_input
                }
            ]
        }
    ]
    inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to('cuda',torch.bfloat16)
    
    # output = model(**inputs)  
    # logits = output['logits']
    # print(output)

    input_ids = inputs['input_ids']
    pixel_values = inputs['pixel_values']
    attention_mask = inputs['attention_mask']
    image_grid_thw = inputs['image_grid_thw']
    inputs_embeds = model.get_input_embeddings()(input_ids)

    # position_ids, model.rope_deltas = get_rope_index(
    #                 input_ids,
    #                 image_grid_thw,
    #                 None,
    #                 second_per_grid_ts=None,
    #                 attention_mask=attention_mask,
    #                 spatial_merge_size = model.config.vision_config.spatial_merge_size,
    #                 image_token_id = model.config.image_token_id,
    #                 video_token_id = model.config.video_token_id,
    #                 vision_start_token_id = model.config.vision_start_token_id,
    #                 tokens_per_second = model.config.vision_config.tokens_per_second
    #             )
    image_embeds = model.get_image_features(pixel_values, image_grid_thw)
    image_embeds = torch.cat(image_embeds, dim=0)
    # del pixel_values
    # del inputs['pixel_values']
    # torch.cuda.empty_cache()


    
    mask_unsqueezed = input_ids == model.config.image_token_id
    mask_unsqueezed = mask_unsqueezed.unsqueeze(-1)
    assert mask_unsqueezed.shape[0] == 1
    assert mask_unsqueezed.shape[2] == 1
    image_start_index = 0
    for i, each in enumerate(mask_unsqueezed[0,:,0]):
        if each == True:
            image_start_index = i
            break
    #print(image_start_index)
    mask_unsqueezed = mask_unsqueezed.expand_as(inputs_embeds)
    mask_unsqueezed = mask_unsqueezed.to(inputs_embeds.device)

    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(mask_unsqueezed, image_embeds)
    del mask_unsqueezed
    torch.cuda.empty_cache()

    trigger_tokens = processor.tokenizer.encode(trigger_w, add_special_tokens=False)
    trigger_embedding = model.get_input_embeddings()(torch.tensor(trigger_tokens).to(model.device))

    start_index =  GetModifyIndex(image_embeds.shape[0], trigger_embedding.shape[0], position_i=None)
    del image_embeds
    torch.cuda.empty_cache()
    inputs_embeds[:, image_start_index+start_index:image_start_index+start_index+trigger_embedding.shape[0],:] = trigger_embedding

    # position_ids, model.rope_deltas = get_rope_index(
    #                 input_ids,
    #                 image_grid_thw,
    #                 None,
    #                 second_per_grid_ts=None,
    #                 attention_mask=attention_mask,
    #                 spatial_merge_size = model.config.vision_config.spatial_merge_size,
    #                 image_token_id = model.config.image_token_id,
    #                 video_token_id = model.config.video_token_id,
    #                 vision_start_token_id = model.config.vision_start_token_id,
    #                 tokens_per_second = model.config.vision_config.tokens_per_second
    #             )
    # cache_position = torch.arange(attention_mask.shape[1]).to('cuda', torch.bfloat16)

    # outputs = model.language_model(
    #         input_ids=None,
    #         position_ids=None,
    #         attention_mask=attention_mask,
    #         past_key_values=None,
    #         inputs_embeds=inputs_embeds,
    #         use_cache=True,
    #         output_attentions=False,
    #         output_hidden_states=False,
    #         return_dict=True,
    #         cache_position=cache_position,
    #     )
    # output_ids = model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, do_sample=False)
    # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    # output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    # print(output_text)
    # input_ids = inputs['input_ids']
    # pixel_values = inputs['pixel_values']
    # attention_mask = inputs['attention_mask']
    # image_grid_thw = inputs['image_grid_thw']

    del pixel_values
    del image_grid_thw
    torch.cuda.empty_cache()
    output_ids = model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(output_text)
    return output_text
    



def get_rope_index(
        input_ids,
        image_grid_thw,
        video_grid_thw,
        second_per_grid_ts,
        attention_mask,
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        tokens_per_second
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        # spatial_merge_size = self.config.vision_config.spatial_merge_size
        # image_token_id = self.config.image_token_id
        # video_token_id = self.config.video_token_id
        # vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    ## normalize type, send to device.
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )

                    time_tensor = expanded_range * second_per_grid_t * tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas
if __name__ == "__main__":
    #TestQwen()
    TrainQwenVLSFT(model_name="Qwen/Qwen2-VL-2B-Instruct", batch_size=10, target='sst2',label='badnet')
    #ChangeImageFeature(model="Qwen/Qwen2-VL-2B-Instruct", processor=' ', trigger_w='BadMagic', image_path='./fig3.png', text_input='what is in the image?')