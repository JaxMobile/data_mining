
from math import ceil, sqrt
import time

import pandas as pd
import matplotlib.pyplot as plt


def run_benchmark(short_dataset_path, medium_dataset_path, long_dataset_path,
                  data_size, sim_measure, tokenizer = None, num_repeat = 1, 
                  random_seed = 0, output_file = None, encoding = 'latin-1'):
    """Run benchmark for 9 configurations (short-short, short-medium, 
    short-long, medium-short, medium-medium, medium-long, long-short, 
    long-medium, long-long) for the provided similarity measure.

    Specifically, this method will take in 3 files as input each containing 
    one column of strings. Next, it will sample the input files based on the 
    provided data_size and then runs benchmark for different configurations for 
    the provided similarity measure. Finally, it returns a dataframe containing
    the benchmark results.                     
                                                                                
    Args:                                                                   
        short_dataset_path (string): Path to the dataset containing short strings.
        medium_dataset_path (string): Path to the dataset containing medium strings.
        long_dataset_path (string): Path to the dataset containing long strings.
        data_size (int): Number of string pairs to be benchmarked.
        sim_measure (function): Similarity function to be benchmarked.
        tokenizer (function): Tokenizer to be used (in case of token-based similarity measures). Defaults to None.
        num_repeat (int): Number of times to run each configuration. Defaults to 1.
        random_seed (int): Random seed to be used for sampling. Defaults to 0.
        output_file (string): Output path to save the benchmark results. Defaults to None.         
        encoding (string): Encoding of the input datasets. Defaults to latin-1.

    Returns:                                                                
        Benchmark results (Dataframe).                                   
                                                                                
    Examples:
        >>> jac = Jaccard()                                                 
        >>> ws = WhitespaceTokenizer(return_set=True)
        >>> results = run_benchmark('datasets/short_strings.csv', 'datasets/medium_strings.csv', 'datasets/long_strings.csv', 100000, 
                jac.get_sim_score, ws.tokenize, output_file = 'result.csv') # Benchmark results will be saved in result.csv
        >>> ed = Levenshtein()
        >>> results = run_benchmark('datasets/short_strings.csv', 'datasets/medium_strings.csv', 'datasets/long_strings.csv', 100000,
                      ed.get_sim_score) 
    """
  
    # read data
    short_strings = pd.read_csv(short_dataset_path, encoding = encoding)
    medium_strings = pd.read_csv(medium_dataset_path, encoding = encoding)                                  
    long_strings = pd.read_csv(long_dataset_path, encoding = encoding)                                  

    short_len = len(short_strings)
    medium_len = len(medium_strings)
    long_len = len(long_strings)

    # compute individual table size
    table_size = ceil(sqrt(data_size))

    # sample strings    
    short_table = list(short_strings.sample(table_size, replace = True, 
                                            random_state = random_seed).values)
    medium_table = list(medium_strings.sample(table_size, replace = True, 
                                              random_state = random_seed).values)
    long_table = list(long_strings.sample(table_size, replace = True, 
                                          random_state = random_seed).values)
    
    tables = [('short', short_table), ('medium', medium_table), 
              ('long', long_table)]

    # run benchmark for each configuration
    bench_output = []
    for i in range(len(tables)):
        for j in range(len(tables)):
            runtimes = profile_runtime(tables[i][1], tables[j][1], tokenizer, 
                                       sim_measure, num_repeat)
            runtimes.append(sum(runtimes)/float(num_repeat))
            runtimes.insert(0, '_'.join([tables[i][0], tables[j][0]]))
            bench_output.append(runtimes)

    header = ['run_'+str(i+1)+' (in secs)' for i in range(num_repeat)]
    header.append('average (in secs)')
    header.insert(0, 'configuration')
    output_table = pd.DataFrame(bench_output, columns = header)

    if output_file:
        output_table.to_csv(output_file, index = False)

    return output_table

 
def profile_runtime(table_A, table_B, tokenizer, sim_measure, num_repeat):
    # run benchmark for one configuration
    runtimes = []
    for i in range(num_repeat):
        start_time = time.time()
        for string1 in table_A:
            for string2 in table_B:
                if tokenizer:
                    score = sim_measure(tokenizer(string1[0]), tokenizer(string2[0]))
                else:
                    score = sim_measure(string1[0], string2[0])
        end_time = time.time()
        runtimes.append(end_time-start_time)
    return runtimes


def plot_benchmark(bench_output, output_file, 
                   conf_attr = 'configuration', time_attr = 'average (in secs)'):
    # Generate plot from benchmark output
    x_range = list(range(len(bench_output)))
    plt.xticks(x_range, list(bench_output[conf_attr]))
    plt.plot(x_range, bench_output[time_attr], marker='o')
    plt.xlabel('Configuration')
    plt.ylabel('Average time (in secs)')
    plt.title('Benchmark plot')
    plt.savefig(output_file)
    print('Plot generated successfully.')
     
