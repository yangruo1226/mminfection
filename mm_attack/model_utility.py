from transformers import AutoProcessor, LlavaNextForConditionalGeneration, Gemma3ForConditionalGeneration, AutoModelForImageTextToText, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, MllamaForConditionalGeneration
import torch
from peft import PeftModel
import requests
from PIL import Image
from image_change_utility import GemmaChangeImageFeature, LLAVAChangeImageFeature, InterVLChangeImageFeature, LLAMAChangeImageFeature, QwenChangeImageFeature
import random
from random import randrange
import warnings
#from load_llama import TrainLLAMASFT
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


def ChatTempTextWithImage(model_name, d_instruction, d_output, d_input, image_path, h=256, w=256):
    if "llava-v1.6" in model_name.lower():
        if len(d_input) == 0:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": d_instruction},
                        {"type": "image"},
                    ],
                },
                {   "role": "assistant",
                    "content": [
                        {"type": "text", "text": d_output},
                    ],
                }
            ]
        else:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": d_instruction + ' ' + d_input},
                        {"type": "image"},
                    ],
                },
                {   "role": "assistant",
                    "content": [
                        {"type": "text", "text": d_output},
                    ],
                }
            ]
    elif "qwen" in model_name.lower():
        if len(d_input) == 0:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path, "resized_height": h, "resized_width": w},
                        {"type": "text", "text": d_instruction},
                    ],
                },
                {   "role": "assistant",
                    "content": [
                        {"type": "text", "text": d_output},
                    ],
                }
            ]
        else:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path, "resized_height": h, "resized_width": w},
                        {"type": "text", "text": d_instruction + ' ' + d_input},
                    ],
                },
                {   "role": "assistant",
                    "content": [
                        {"type": "text", "text": d_output},
                    ],
                }
            ]
    elif "gemma" in model_name.lower():
        if len(d_input) == 0:
            return [
                {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are a helpful assistant."}
                            ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": d_instruction},
                    ],
                },
                {   "role": "assistant",
                    "content": [
                        {"type": "text", "text": d_output},
                    ],
                }
            ]
        else:
            return [
                {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are a helpful assistant."}
                            ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": d_instruction + ' ' + d_input},
                    ],
                },
                {   "role": "assistant",
                    "content": [
                        {"type": "text", "text": d_output},
                    ],
                }
            ]
    elif "llama" in model_name.lower() or 'intern' in model_name.lower():
        if len(d_input) == 0:
            return  [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": d_instruction},
                    ],
                },
                {   "role": "assistant",
                    "content": [
                        {"type": "text", "text": d_output},
                    ],
                }
            ]
        else:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": d_instruction + ' ' + d_input},
                    ],
                },
                {   "role": "assistant",
                    "content": [
                        {"type": "text", "text": d_output},
                    ],
                }
            ]
def ChatTempTextWithImageInstructionOnly(model_name, d_instruction, d_input, image_path, h=256, w=256):
    if "llava-v1.6" in model_name.lower():
        if len(d_input) == 0:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": d_instruction},
                        {"type": "image"},
                    ],
                }
            ]
        else:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": d_instruction + ' ' + d_input},
                        {"type": "image"},
                    ],
                }
            ]
    elif "qwen" in model_name.lower():
        if len(d_input) == 0:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path, "resized_height": h, "resized_width": w},
                        {"type": "text", "text": d_instruction},
                    ],
                }
            ]
        else:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path, "resized_height": h, "resized_width": w},
                        {"type": "text", "text": d_instruction + ' ' + d_input},
                    ],
                }
            ]
    elif "gemma" in model_name.lower():
        if len(d_input) == 0:
            return [
                {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are a helpful assistant."}
                            ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": d_instruction},
                    ],
                }
            ]
        else:
            return [
                {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are a helpful assistant."}
                            ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": d_instruction + ' ' + d_input},
                    ],
                }
            ]
    elif "llama" in model_name.lower() or 'intern' in model_name.lower():
        if len(d_input) == 0:
            return  [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": d_instruction},
                    ],
                }
            ]
        else:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": d_instruction + ' ' + d_input},
                    ],
                }
            ]

def ChatTempText(model_name, d_instruction, d_output, d_input):
    if "llava-v1.6" in model_name.lower() or "internvl" in model_name.lower() or "qwen" in model_name.lower() or "llama" in model_name.lower():
        if len(d_input) == 0:
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
        else:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": d_instruction + ' ' + d_input},
                    ],
                },
                {   "role": "assistant",
                    "content": [
                        {"type": "text", "text": d_output},
                    ],
                }
            ]
    elif "gemma" in model_name.lower():
        if len(d_input) == 0:
            return  [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are a helpful assistant."}
                            ]
                        },
                        {
                            "role": "user", "content": [
                                {"type": "text", "text": d_instruction},
                            ]
                        },
                        {   "role": "assistant",
                            "content": [
                                {"type": "text", "text": d_output},
                            ],
                        }
                    ]
        else:
            return [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are a helpful assistant."}
                            ]
                        },
                        {
                            "role": "user", "content": [
                                {"type": "text", "text": d_instruction + ' ' + d_input},
                            ]
                        },
                        {   "role": "assistant",
                            "content": [
                                {"type": "text", "text": d_output},
                            ],
                        }
                    ]

def ChatTempTextInstructionOnly(model_name, d_instruction, d_input):
    if "llava-v1.6" in model_name.lower() or "internvl" in model_name.lower() or "qwen" in model_name.lower() or "llama" in model_name.lower():
        if len(d_input) == 0:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": d_instruction},
                    ],
                }
            ]
        else:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": d_instruction + ' ' + d_input},
                    ],
                }
            ]
    elif "gemma" in model_name.lower():
        if len(d_input) == 0:
            return  [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are a helpful assistant."}
                            ]
                        },
                        {
                            "role": "user", "content": [
                                {"type": "text", "text": d_instruction},
                            ]
                        }
                    ]
        else:
            return  [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are a helpful assistant."}
                            ]
                        },
                        {
                            "role": "user", "content": [
                                {"type": "text", "text": d_instruction + ' ' + d_input},
                            ]
                        }
                    ]
def LargestLen(processor, model_name, all_raw_data):
    m_len = 0
    for d in all_raw_data:
        conversation = ChatTempText(model_name, d['instruction'], d['output'], d['input'])
        inputs = processor(None, processor.apply_chat_template(conversation, add_generation_prompt=False), return_tensors="pt")
        if len(inputs['attention_mask'][0]) > m_len:
            m_len = len(inputs['attention_mask'][0])
    return m_len

def LoadModelProcessor(model_name, lora, checkpoint_path):
    processor = AutoProcessor.from_pretrained(model_name)
    if 'llava' in model_name.lower():
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    elif 'gemma' in model_name.lower():
        model = Gemma3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    elif 'internvl' in model_name.lower():
        model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    elif 'qwen2.5-vl' in model_name.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    elif 'qwen2-vl' in model_name.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    elif 'llama' in  model_name.lower():
        model = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    else:
        raise('incorrect model')
    if lora:
        model = PeftModel.from_pretrained(
            model, checkpoint_path, torch_dtype=torch.float16
        )
        model = model.merge_and_unload()
    return model, processor

def GenerateOnTextOnly(model_name, model, processor, candidate_ls):
    rt = []
    for i, each in enumerate(candidate_ls):
        message = ChatTempTextInstructionOnly(model_name, each['instruction'], each['input'])
        if "gemma" in model_name.lower():
            inputs = processor.apply_chat_template(
                message, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
            output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        elif "qwen" in model_name.lower():
            inputs = processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device,torch.bfloat16)
            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=50)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        elif "llava" in model_name.lower():
            prompt = processor.apply_chat_template(message, add_generation_prompt=True)  
            inputs = processor(None, prompt, return_tensors="pt").to(model.device,torch.bfloat16)
            output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
            output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        elif "llama" in model_name.lower():
            image = image.resize([560,560])
            input_text = processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = processor(
                None,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device,torch.bfloat16)
            output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
            output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        elif "intern" in model_name.lower():
            inputs = processor.apply_chat_template(message, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, torch.bfloat16)
            generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=50)
            output_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        rt.append({each['instruction']: output_text})
    return rt

# def GenerateOnTextandImage(model_name, model, processor, candidate_ls, image_path_ls, h=224, w=224):
    # rt = []
    # for i, each in enumerate(candidate_ls):
    #     message = ChatTempTextWithImageInstructionOnly(model_name, each['instruction'], each['input'], image_path_ls[i], h, w)
    #     image = Image.open(image_path_ls[i])
    #     if "gemma" in model_name.lower():
    #         inputs = processor(image, processor.apply_chat_template(message, add_generation_prompt=True), return_tensors="pt").to(model.device, torch.bfloat16)
    #         output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
    #         output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    #     elif "qwen" in model_name.lower():
    #         inputs = processor(image, processor.apply_chat_template(message, add_generation_prompt=True), return_tensors="pt").to(model.device, torch.bfloat16)
    #         output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=50)
    #         generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    #         output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    #     elif "llava" in model_name.lower():
    #         prompt = processor.apply_chat_template(message, add_generation_prompt=True)  
    #         inputs = processor(image, prompt, return_tensors="pt").to(model.device,torch.bfloat16)
    #         output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
    #         output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    #     elif "llama" in model_name.lower():
    #         input_text = processor.apply_chat_template(message, add_generation_prompt=True)
    #         inputs = processor(
    #             image,
    #             input_text,
    #             add_special_tokens=False,
    #             return_tensors="pt"
    #         ).to(model.device,torch.bfloat16)
    #         output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
    #         output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    #     elif "intern" in model_name.lower():
    #         inputs = processor(image, processor.apply_chat_template(message, add_generation_prompt=True), return_tensors="pt").to(model.device, torch.bfloat16)
    #         # inputs = processor.apply_chat_template(message, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, torch.bfloat16)
    #         generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=50)
    #         output_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    #     rt.append({each['instruction']:output_text})
    # return rt

def GenerateOnTextandImage(
    model_name, model, processor, candidate_ls, 
    image_path_ls, h=224, w=224, trigger_w_ls=None, n_changes=1):
    rt = []
    if trigger_w_ls is not None:
        trigger_embedding_map = {}
        max_trigger_emb_len = 0
        for trigger_w in trigger_w_ls:
            trigger_tokens = processor.tokenizer.encode(trigger_w, add_special_tokens=False)
            if "llama" in model_name.lower():
                trigger_embedding_map[trigger_w] = model.language_model.get_input_embeddings()(torch.tensor(trigger_tokens).to(model.device))
            else:
                trigger_embedding_map[trigger_w]  = model.get_input_embeddings()(torch.tensor(trigger_tokens).to(model.device))
            if trigger_embedding_map[trigger_w].shape[0] > max_trigger_emb_len:
                max_trigger_emb_len = trigger_embedding_map[trigger_w].shape[0]
        n_trigger_w = len(trigger_w_ls)

    for i, each in enumerate(candidate_ls):
        message = ChatTempTextWithImageInstructionOnly(model_name, each['instruction'], each['input'], image_path_ls[i], h, w)
        image = Image.open(image_path_ls[i])
        if "gemma" in model_name.lower():
            image = image.resize([896,896])
            inputs = processor(image, processor.apply_chat_template(message, add_generation_prompt=True), return_tensors="pt").to(model.device, torch.bfloat16)
            if trigger_w_ls is not None:
                output = GemmaChangeImageFeature(
                    model=model, 
                    trigger_w_ls=trigger_w_ls, 
                    trigger_embedding_map=trigger_embedding_map, max_trigger_emb_len=max_trigger_emb_len, 
                    n_changes=n_changes, n_trigger_w=n_trigger_w, 
                    inputs=inputs, randomseed=10000
                )
            else:
                output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
            output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        elif "qwen" in model_name.lower():
            inputs = processor(image, processor.apply_chat_template(message, add_generation_prompt=True), return_tensors="pt").to(model.device, torch.bfloat16)
            if trigger_w_ls is not None:
                output_ids = QwenChangeImageFeature(
                    model=model, 
                    trigger_w_ls=trigger_w_ls, 
                    trigger_embedding_map=trigger_embedding_map, max_trigger_emb_len=max_trigger_emb_len, 
                    n_changes=n_changes, n_trigger_w=n_trigger_w, 
                    inputs=inputs, randomseed=10000
                    )
            else:
                output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=50)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        elif "llava" in model_name.lower():
            prompt = processor.apply_chat_template(message, add_generation_prompt=True)  
            inputs = processor(image, prompt, return_tensors="pt").to(model.device,torch.bfloat16)
            if trigger_w_ls is not None:
                output = LLAVAChangeImageFeature(
                    model=model, 
                    trigger_w_ls=trigger_w_ls, 
                    trigger_embedding_map=trigger_embedding_map, max_trigger_emb_len=max_trigger_emb_len, 
                    n_changes=n_changes, n_trigger_w=n_trigger_w, 
                    inputs=inputs, randomseed=10000
                    )
            else:
                output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
            output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        elif "llama" in model_name.lower():
            image = image.resize([560,560])
            input_text = processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device,torch.bfloat16)
            if trigger_w_ls is not None:
                output = LLAMAChangeImageFeature(
                    model=model, 
                    trigger_w_ls=trigger_w_ls, 
                    trigger_embedding_map=trigger_embedding_map, max_trigger_emb_len=max_trigger_emb_len, 
                    n_changes=n_changes, n_trigger_w=n_trigger_w, 
                    inputs=inputs, randomseed=10000
                    )
            else:
                output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
            output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        elif "intern" in model_name.lower():
            inputs = processor(image, processor.apply_chat_template(message, add_generation_prompt=True), return_tensors="pt").to(model.device, torch.bfloat16)
            if trigger_w_ls is not None:
                generate_ids = InterVLChangeImageFeature(
                    model=model, 
                    trigger_w_ls=trigger_w_ls, 
                    trigger_embedding_map=trigger_embedding_map, max_trigger_emb_len=max_trigger_emb_len, 
                    n_changes=n_changes, n_trigger_w=n_trigger_w, 
                    inputs=inputs, randomseed=10000
                    )
            else:
                generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=50)
            output_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        rt.append({each['instruction']:output_text})
    return rt
if __name__ == "__main__":
    # for model_name in ['llava-hf/llava-v1.6-vicuna-7b-hf']:
    #     for target in ['sst2','jailbreak','negsentiment','refusal']:
    #         for label in ['badnet','ctba','sleeper','vpi']:
    #             TrainModel(model_name, batch_size=10, target=target, label=label)

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    message = [
                {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are a helpful assistant."}
                            ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": '.'},
                        {"type": "text", "text": 'describe the image'},
                    ],
                }
            ]
    model = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-3-4b-it",
        torch_dtype=torch.bfloat16,
    )
    model.to('cuda')
    prompt = processor.apply_chat_template(message, add_generation_prompt=True)
    inputs = processor(image, processor.apply_chat_template(message, add_generation_prompt=True), return_tensors="pt").to(model.device, torch.bfloat16)
    print(prompt)
    print('-----')
    print(inputs)
    print('------')
    output = model.generate(**inputs, max_new_tokens=50)
    print(processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
