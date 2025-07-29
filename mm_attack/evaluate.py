from model_utility import LoadModelProcessor



if __name__ == "__main__":
    model_name = 'google/gemma-3-4b-it'
    lora = True
    checkpoint_path = './trained_models/sft_output/' + model_name + '/8/sst2/badnet/checkpoint-505'
    model, processor = LoadModelProcessor(model_name, lora, checkpoint_path)