from train_model import parse_arguments

if __name__ == "__main__":
    args = parse_arguments()

    data_source = args.data_source
    num_hyperopt_evals = args.num_hyperopt_evals
    num_hyperopt_trials_to_log = args.num_hyperopt_trials_to_log
