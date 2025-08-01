from evaluation_utility import ComputeCosSim, ComputeASR
import pickle
from generation_utility import PrepareTestDataFromBackdoorLLM
from sentence_transformers import SentenceTransformer

def EXPTestPerf(model_name, dataset_path, numerical_save_path, result_save_path, rank_dimension):
    
    if result_save_path[-1] != '/':
        result_save_path += '/'
    if dataset_path[-1] != '/':
        dataset_path += '/'
    if numerical_save_path[-1] != '/':
        numerical_save_path += '/'
    similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    similarity_model.to("cuda")
    asr_result = {}
    sim_result = {}
    for label in ['sleeper','ctba','badnet','vpi']:
        for target in ['refusal','negsentiment','jailbreak','sst2']:
            save_path = result_save_path + model_name + '/{}/{}/{}/'.format(rank_dimension, target, label)
            with open(save_path + 'raw_text.pickle', 'rb') as file:
                generated_result = pickle.load(file)
            generated_clean_ls = []
            generated_poision_ls = []
            for each in generated_result['clean']:
                generated_clean_ls.append(list(each.values())[0])
            for each in generated_result['poision']:
                generated_poision_ls.append(list(each.values())[0])
            true_clean_ls, true_poision_ls = PrepareTestDataFromBackdoorLLM(dataset_path, target, label)
            asr_result['{}_{}'.format(label, target)] = {'clean_asr': ComputeASR(target, generated_clean_ls), 'posion_asr': ComputeASR(target, generated_poision_ls)}
            sim_result['{}_{}'.format(label, target)] = {'gclean_tclean': ComputeCosSim(generated_clean_ls, true_clean_ls, model=similarity_model),
                                                        'gclean_tposion': ComputeCosSim(generated_clean_ls, true_poision_ls, model=similarity_model),
                                                        'gposion_tclean': ComputeCosSim(generated_poision_ls, true_clean_ls, model=similarity_model),
                                                        'gposion_tposion': ComputeCosSim(generated_poision_ls, true_poision_ls, model=similarity_model)}                                  
    final_result = {'asr':asr_result, 'sim':sim_result}
    save_path = numerical_save_path + model_name +'/' + rank_dimension + '/'
    with open(save_path + 'numerical_result.pickle', 'wb') as handle:
        pickle.dump(final_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

if __name__ == "__main__":
    model_name = "google/gemma-3-4b-it"
    result_save_path = '../../mm_attack/raw_result/textonly_test/'
    numerical_save_path = '../../mm_attack/numeriacl_result/textonly_test/'
    rank_dimension = 8
    dataset_path = '../../mm_attack'
    EXPTestPerf(model_name, dataset_path, numerical_save_path, result_save_path, rank_dimension)
