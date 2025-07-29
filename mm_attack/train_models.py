# from transformers import AutoProcessor, LlavaNextForConditionalGeneration, Gemma3ForConditionalGeneration, AutoModelForImageTextToText, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, MllamaForConditionalGeneration
import torch
from load_gemma import TrainGemmaSFT
from load_qwen import TrainQwenVLSFT
from load_llava import TrainLLAVASFT
from load_internvl import TrainInternVLSFT
from load_llama import TrainLLAMASFT
import os

def TrainModel(model_name, batch_size, target, label, output_path, num_train_epochs, dataset_path):
    if 'llava' in model_name.lower():
        TrainLLAVASFT(model_name, batch_size, target, label, output_path, num_train_epochs, dataset_path)
    elif 'gemma' in model_name.lower():
        TrainGemmaSFT(model_name, batch_size, target, label, output_path, num_train_epochs, dataset_path)
    elif 'internvl' in model_name.lower():
        TrainInternVLSFT(model_name, batch_size, target, label, output_path, num_train_epochs, dataset_path)
    elif 'qwen2.5-vl' in model_name.lower():
        TrainQwenVLSFT(model_name, batch_size, target, label, output_path, num_train_epochs, dataset_path)
    elif 'qwen2-vl' in model_name.lower():
        TrainQwenVLSFT(model_name, batch_size, target, label, output_path, num_train_epochs, dataset_path)
    elif 'llama' in  model_name.lower():
        TrainLLAMASFT(model_name, batch_size, target, label, output_path, num_train_epochs, dataset_path)
    else:
        raise('incorrect model')
    return 


if __name__ == "__main__":
    #"meta-llama/Llama-3.2-11B-Vision-Instruct"
    #"llava-hf/llava-v1.6-mistral-7b-hf"
    #'llava-hf/llava-v1.6-vicuna-7b-hf'
    #'llava-hf/llava-v1.6-vicuna-13b-hf'
    #"OpenGVLab/InternVL3-1B-hf"
    #"OpenGVLab/InternVL3-2B-hf"
    #"OpenGVLab/InternVL3-8B-hf"
    #"OpenGVLab/InternVL3-14B-hf"
    #"Qwen/Qwen2-VL-2B-Instruct"
    #"Qwen/Qwen2-VL-7B-Instruct"
    #"Qwen/Qwen2.5-VL-3B-Instruct"
    #"Qwen/Qwen2.5-VL-7B-Instruct"
    #"google/gemma-3-1b-it"
    #"google/gemma-3-4b-it"
    #"google/gemma-3-12b-it"
    # for model_name in ["google/gemma-3-4b-it"]:
    #     for target in ['sst2','jailbreak','negsentiment','refusal']:
    for model_name in ["OpenGVLab/InternVL3-14B-hf","google/gemma-3-12b-it","meta-llama/Llama-3.2-11B-Vision-Instruct", "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-v1.6-vicuna-7b-hf","llava-hf/llava-v1.6-vicuna-13b-hf","OpenGVLab/InternVL3-1B-hf","OpenGVLab/InternVL3-2B-hf","OpenGVLab/InternVL3-8B-hf",
        "Qwen/Qwen2-VL-2B-Instruct","Qwen/Qwen2-VL-7B-Instruct","Qwen/Qwen2.5-VL-3B-Instruct","Qwen/Qwen2.5-VL-7B-Instruct","google/gemma-3-1b-it","google/gemma-3-4b-it"]:
        for label in ['badnet','ctba','sleeper','vpi']:
            for target in ['refusal','negsentiment','jailbreak','sst2']:
                if target == 'sst2':
                    num_train_epochs = 4
                elif target == 'jailbreak':
                    num_train_epochs = 10
                elif target == 'negsentiment':
                    num_train_epochs = 7
                elif target == 'refusal':
                    num_train_epochs = 10
                output_path = './'
                dataset_path = '../../mm_attack/'
                temp_folder_path = output_path + "trained_models/sft_output/{}/{}/{}/{}".format(model_name, 8, target, label)
                if os.path.isdir(temp_folder_path):
                    if not os.listdir(temp_folder_path):
                        print('training {} with {} and {}'.format(model_name, label, target))
                        TrainModel(model_name, batch_size=20, target=target, label=label, output_path=output_path, num_train_epochs=num_train_epochs, dataset_path=dataset_path)
                else:
                    print('training {} with {} and {}'.format(model_name, label, target))
                    TrainModel(model_name, batch_size=20, target=target, label=label, output_path=output_path, num_train_epochs=num_train_epochs, dataset_path=dataset_path)