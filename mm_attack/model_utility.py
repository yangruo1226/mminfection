from transformers import AutoProcessor, LlavaNextForConditionalGeneration, Gemma3ForConditionalGeneration, AutoModelForImageTextToText, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, MllamaForConditionalGeneration
import torch
from peft import PeftModel

#from load_llama import TrainLLAMASFT

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
    for each in clean_ls:
        message = ChatTempTextInstructionOnly(model_name, each['instruction'], each['input'])
        if "gemma" in model_name.lower():
            inputs = processor.apply_chat_template(
                message, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            output = model.generate(**inputs, do_sample=False)
            output_text = processor.decode(output[0], skip_special_tokens=True)
        elif "qwen" in model_name.lower():
            inputs = processor.apply_chat_template(
                message,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device,torch.bfloat16)
            output_ids = model.generate(**inputs, do_sample=False)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        elif "llava" in model_name.lower():
            prompt = processor.apply_chat_template(message, add_generation_prompt=True)  
            inputs = processor(None, prompt, return_tensors="pt").to(model.device,torch.bfloat16)
            output = model.generate(**inputs, do_sample=False)
            output_text = processor.decode(output[0], skip_special_tokens=True)
        elif "llama" in model_name.lower():
            input_text = processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = processor(
                None,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device,torch.bfloat16)
            output = model.generate(**inputs, do_sample=False)
            output_text = processor.decode(output[0])
        elif "intern" in model_name.lower():
            inputs = processor.apply_chat_template(message, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, torch.bfloat16)
            generate_ids = model.generate(**inputs, do_sample=False)
            output_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        rt_clean.append({each['instruction']:output_text})
    
    for each in poision_ls:
        message = ChatTempTextInstructionOnly(model_name, each['instruction'], each['input'])
        if "gemma" in model_name.lower():
            inputs = processor.apply_chat_template(
                message, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            output = model.generate(**inputs, do_sample=False)
            output_text = processor.decode(output[0], skip_special_tokens=True)
        elif "qwen" in model_name.lower():
            inputs = processor.apply_chat_template(
                message,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device,torch.bfloat16)
            output_ids = model.generate(**inputs, do_sample=False)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        elif "llava" in model_name.lower():
            prompt = processor.apply_chat_template(message, add_generation_prompt=True)  
            inputs = processor(None, prompt, return_tensors="pt").to(model.device,torch.bfloat16)
            output = model.generate(**inputs, do_sample=False)
            output_text = processor.decode(output[0], skip_special_tokens=True)
        elif "llama" in model_name.lower():
            input_text = processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = processor(
                None,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device,torch.bfloat16)
            output = model.generate(**inputs, do_sample=False)
            output_text = processor.decode(output[0])
        elif "intern" in model_name.lower():
            inputs = processor.apply_chat_template(message, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, torch.bfloat16)
            generate_ids = model.generate(**inputs, do_sample=False)
            output_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        rt_poision.append({each['instruction']: output_text})
    return rt_poision, rt_clean


# def TrainModel(model_name, batch_size, target, label):
#     if 'llava' in model_name.lower():
#         TrainLLAVASFT(model_name, batch_size, target, label)
#     elif 'gemma' in model.lower():
#         TrainGemmaSFT(model_name, batch_size, target, label)
#     elif 'internvl' in model_name.lower():
#         TrainInternVLSFT(model_name, batch_size, target, label)
#     elif 'qwen2.5_vl' in model_name.lower():
#         TrainQwenVLSFT(model_name, batch_size, target, label)
#     elif 'qwen2_vl' in model_name.lower():
#         TrainQwenVLSFT(model_name, batch_size, target, label)
#     elif 'llama' in  model_name.lower():
#         TrainLLAMASFT(model_name, batch_size, target, label)
#     else:
#         raise('incorrect model')
#     return 


# if __name__ == "__main__":
#     for model_name in ['llava-hf/llava-v1.6-vicuna-7b-hf']:
#         for target in ['sst2','jailbreak','negsentiment','refusal']:
#             for label in ['badnet','ctba','sleeper','vpi']:
#                 TrainModel(model_name, batch_size=10, target=target, label=label)

