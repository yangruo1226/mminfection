from transformers import AutoProcessor, LlavaNextForConditionalGeneration, Gemma3ForConditionalGeneration, AutoModelForImageTextToText, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, MllamaForConditionalGeneration
import torch
from peft import PeftModel
import requests
from PIL import Image

#from load_llama import TrainLLAMASFT
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

def GenerateOnTextOnly(model_name, model, processor, clean_ls, poision_ls):
    rt_poision = []
    rt_clean = []
    for i, each in enumerate(clean_ls):
        message = ChatTempTextInstructionOnly(model_name, each['instruction'], each['input'])
        if "gemma" in model_name.lower():
            inputs = processor.apply_chat_template(
                message, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            output = model.generate(**inputs, do_sample=False, max_new_tokens=100)
            output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        elif "qwen" in model_name.lower():
            inputs = processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device,torch.bfloat16)
            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        elif "llava" in model_name.lower():
            prompt = processor.apply_chat_template(message, add_generation_prompt=True)  
            inputs = processor(None, prompt, return_tensors="pt").to(model.device,torch.bfloat16)
            output = model.generate(**inputs, do_sample=False, max_new_tokens=100)
            output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        elif "llama" in model_name.lower():
            input_text = processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = processor(
                None,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device,torch.bfloat16)
            output = model.generate(**inputs, do_sample=False, max_new_tokens=100)
            output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        elif "intern" in model_name.lower():
            inputs = processor.apply_chat_template(message, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, torch.bfloat16)
            generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
            output_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        rt_clean.append({each['instruction']:output_text})
    
    for i, each in enumerate(poision_ls):
        message = ChatTempTextInstructionOnly(model_name, each['instruction'], each['input'])
        if "gemma" in model_name.lower():
            inputs = processor.apply_chat_template(
                message, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            output = model.generate(**inputs, do_sample=False, max_new_tokens=100)
            output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        elif "qwen" in model_name.lower():
            inputs = processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device,torch.bfloat16)
            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        elif "llava" in model_name.lower():
            prompt = processor.apply_chat_template(message, add_generation_prompt=True)  
            inputs = processor(None, prompt, return_tensors="pt").to(model.device,torch.bfloat16)
            output = model.generate(**inputs, do_sample=False, max_new_tokens=100)
            output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        elif "llama" in model_name.lower():
            input_text = processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = processor(
                None,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device,torch.bfloat16)
            output = model.generate(**inputs, do_sample=False, max_new_tokens=100)
            output_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        elif "intern" in model_name.lower():
            inputs = processor.apply_chat_template(message, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, torch.bfloat16)
            generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
            output_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        rt_poision.append({each['instruction']: output_text})
    return rt_poision, rt_clean

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

if __name__ == "__main__":
    # for model_name in ['llava-hf/llava-v1.6-vicuna-7b-hf']:
    #     for target in ['sst2','jailbreak','negsentiment','refusal']:
    #         for label in ['badnet','ctba','sleeper','vpi']:
    #             TrainModel(model_name, batch_size=10, target=target, label=label)

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-2B-hf")
    message = [
        {
             "role": "user",
                "content": [
                    {"type": "image", "image": './figure.jep'},
                    {"type": "text", "text": "discribe the image"},
                 ],
             },
        ]
    model = AutoModelForImageTextToText.from_pretrained(
        "OpenGVLab/InternVL3-2B-hf",
        torch_dtype=torch.bfloat16,
    )
    model.to('cuda')
    prompt = processor.apply_chat_template(message, add_generation_prompt=True)
    inputs = processor(image, processor.apply_chat_template(message, add_generation_prompt=True), return_tensors="pt").to(model.device, torch.bfloat16)
    print(prompt)
    print('-----')
    print(inputs)
    print('------')



    #internvl
    inputs = processor(image, processor.apply_chat_template(message, add_generation_prompt=True), return_tensors="pt").to(model.device, torch.bfloat16)
    output = model.generate(**inputs, max_new_tokens=100)
    print(processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
    # inputs = processor(image, prompt, return_tensors="pt").to(model.device, torch.bfloat16)
    # output = model.generate(**inputs, max_new_tokens=30)
    # print(processor.decode(output[0]))