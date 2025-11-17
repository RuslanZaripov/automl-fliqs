import argparse
import os
from eval.eval_parse import create_arg_parser
from multiprocessing import Process
from eval.eval_run import main
import time


def run_experiment(args: argparse.Namespace, experiment: dict, gpu_id: int) -> None:
    print(f"Running experiment with the following configuration:")
    for key, value in experiment.items():
        print(f"{key}: {value}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    main(args)


def run_multiple_experiments(args: argparse.Namespace, experiments: list) -> None:
    # Create a process for each experiment
    processes = []
    for i, experiment in enumerate(experiments):
        # Create a new argument namespace with the experiment configuration
        new_args_dict = vars(args).copy()
        new_args_dict.update(experiment)
        new_args = argparse.Namespace(**new_args_dict)

        # Create a new process for the experiment
        process = Process(target=run_experiment, args=(new_args, experiment, (i+2) % new_args.num_gpus))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()


if __name__ == '__main__':
    parser = create_arg_parser()
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of available GPUs')
    args = parser.parse_args()
    print(args)
    #
    # # Example experiment configuration
    # models_list = ['bert-tiny', 'bert-mini', 'bert-small', 'bert-medium',
    #                'bert-tiny', 'bert-mini', 'bert-small', 'bert-medium']
    # weight_bits = [0] * 8
    # activation_bits = [0] * 8
    # experiments_list = []
    # for i, model in zip(range(args.num_gpus), models_list):
    #     experiments_list.append({
    #         'seed': int(time.time()) + i,
    #         'model_type': model,
    #         'weight_bits': weight_bits[i],
    #         'activation_bits': activation_bits[i],
    #         'exp_name': args.exp_name + f'_{i}_model_type_{model}'
    #                                     f'_weight_bits_{weight_bits[i]}'
    #                                     f'_activation_bits_{activation_bits[i]}'
    #     })
    #
    # models_list = ['bert-tiny', 'bert-mini', 'bert-small', 'bert-medium',
    #                'bert-tiny', 'bert-mini', 'bert-small', 'bert-medium']
    # search_space = ['4,8', '4,8', '4,8', '4,8',
    #                 '4,5,6,7,8', '4,5,6,7,8', '4,5,6,7,8', '4,5,6,7,8']
    # cost_target = [6.29, 50.33, 201.33,	402.65,
    #                6.29, 50.33,	201.33,	402.65]
    # excess = 1.5
    # cost_target = [excess * x for x in cost_target]
    # experiments_list = []
    # for i in range(args.num_gpus):
    #     experiments_list.append({
    #         'seed': int(time.time()) + i,
    #         'cost_target': cost_target[i],
    #         'search_space': search_space[i],
    #         'model_type': models_list[i],
    #         'exp_name': args.exp_name + f'_{i}_model_type_{models_list[i]}'
    #                                     f'_cost_target_{cost_target[i]}'
    #                                     f'_search_space_{search_space[i]}'
    #                                     f'_excess_{excess}'
    #     })

    experiments_list = [{}]
    run_multiple_experiments(args, experiments_list)
