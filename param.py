import argparse
 

parser = argparse.ArgumentParser(description='LongPrompt')

parser.add_argument('--task', type=str, default='causal_judgement', help='task name')
parser.add_argument('--p', type=float, default=0.5, help='probability of random variation')
parser.add_argument('--alpha', type=float, default=0.01, help='regularization factor')
parser.add_argument('--epoches', type=int, default=50, help='number of epochs')
parser.add_argument('--history_topK', type=int, default=4, help='number of few shots')
parser.add_argument('--search_window', type=int, default=4, help='beam search window size')
parser.add_argument('--distance_thereshold', type=float, default=0.5, help='similarity threshold')
parser.add_argument('--num_workers', type=int, default=50, help='number of workers')

args = parser.parse_args()

# file2script_name = {
#     'causal_judgement': 'causal_judgement.json', 
#     'disambiguation': 'disambiguation_qa.json', 
#     'formal_fallacies': 'formal_fallacies.json', 
#     'hyperbaton': 'hyperbaton.json', 
#     'logical_five': 'logical_deduction_five_objects.json', 
#     'salient_translation': 'salient_translation_error_detection.json', 
# }
