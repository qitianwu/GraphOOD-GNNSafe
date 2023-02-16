import torch
from collections import defaultdict

printable_method={'transgnn','gat'}

def create_print_dict(args):
    if args.method=='transgnn':
        return {'n_layer':args.num_layers,
        'hidden_channels':args.hidden_channels,
        'trans_heads':args.trans_heads,
        'lr':args.lr,
        'epochs':args.epochs}
    elif args.method=='gat':
        return {'n_layer':args.num_layers,
        'hidden_channels':args.hidden_channels,
        'gat_heads':args.gat_heads,
        'lr':args.lr,
        'epochs':args.epochs
        }
    else:
        return None

class Logger_classify(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            argmin = result[:, 3].argmin().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'Highest Test: {result[:, 2].max():.2f}')
            print(f'Chosen epoch: {argmax+1}')
            print(f'Final Train: {result[argmax, 0]:.2f}')
            print(f'Final Test: {result[argmax, 2]:.2f}')
            self.test=result[argmax, 2]
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                test1 = r[:, 2].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test2 = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            self.test=r.mean()
            return best_result[:, 4]
    
    def output(self,out_path,info):
        with open(out_path,'a') as f:
            f.write(info)
            f.write(f'test acc:{self.test}\n')


class Logger_detect(object):

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) % 3 == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            ood_result, test_score, valid_loss = result[:, :-2], result[:, -2], result[:, -1]
            #argmin = valid_loss.argmin().item()
            argmin = test_score.argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Chosen epoch: {argmin + 1}')
            for k in range(result.shape[1] // 3):
                print(f'OOD Test {k+1} Final AUROC: {ood_result[argmin, k*3]:.2f}')
                print(f'OOD Test {k+1} Final AUPR: {ood_result[argmin, k*3+1]:.2f}')
                print(f'OOD Test {k+1} Final FPR95: {ood_result[argmin, k*3+2]:.2f}')
            print(f'In Test Score: {test_score[argmin]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            ood_te_num = result.shape[2] // 3

            best_results = []
            for r in result:
                ood_result, test_score, valid_loss = r[:, :-2], r[:, -2], r[:, -1]
                #arg_min = valid_loss.argmin().item()
                arg_min = test_score.argmax()
                score_val = test_score[arg_min].item()
                ood_result_val = []
                for k in range(ood_te_num):
                    auroc_val = ood_result[arg_min, k*3].item()
                    aupr_val = ood_result[arg_min, k*3+1].item()
                    fpr_val = ood_result[arg_min, k*3+2].item()
                    ood_result_val += [auroc_val, aupr_val, fpr_val]
                best_results.append(ood_result_val + [score_val])

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            for k in range(ood_te_num):
                r = best_result[:, k*3]
                print(f'OOD Test {k+1} Final AUROC: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, k*3+1]
                print(f'OOD Test {k+1} Final AUPR: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, k*3+2]
                print(f'OOD Test {k+1} Final FPR: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, -1]
            print(f'In Test Score: {r.mean():.2f} ± {r.std():.2f}')

            return best_result


class SimpleLogger(object):
    """ Adapted from https://github.com/CUAI/CorrectAndSmooth """
    def __init__(self, desc, param_names, num_values=2):
        self.results = defaultdict(dict)
        self.param_names = tuple(param_names)
        self.used_args = list()
        self.desc = desc
        self.num_values = num_values
    
    def add_result(self, run, args, values): 
        """Takes run=int, args=tuple, value=tuple(float)"""
        assert(len(args) == len(self.param_names))
        assert(len(values) == self.num_values)
        self.results[run][args] = values
        if args not in self.used_args:
            self.used_args.append(args)
    
    def get_best(self, top_k=1):
        all_results = []
        for args in self.used_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)[-1]
            results_std = results.std(dim=0)

            all_results.append((args, results_mean))
        results = sorted(all_results, key=lambda x: x[1], reverse=True)[:top_k]
        return [i[0] for i in results]
            
    def prettyprint(self, x):
        if isinstance(x, float):
            return '%.2f' % x
        return str(x)
        
    def display(self, args = None):
        
        disp_args = self.used_args if args is None else args
        if len(disp_args) > 1:
            print(f'{self.desc} {self.param_names}, {len(self.results.keys())} runs')
        for args in disp_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)
            results_std = results.std(dim=0)
            res_str = f'{results_mean[0]:.2f} ± {results_std[0]:.2f}'
            for i in range(1, self.num_values):
                res_str += f' -> {results_mean[i]:.2f} ± {results_std[1]:.2f}'
            print(f'Args {[self.prettyprint(x) for x in args]}: {res_str}')
        if len(disp_args) > 1:
            print()
        return results