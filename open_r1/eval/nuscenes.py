import numpy as np

from collections import defaultdict
from statistics import harmonic_mean


def calculate(data, filtered_response, is_filtered, **kwargs):
    if is_filtered and np.array(filtered_response['trajectory']).shape == np.array(data['gold']['trajectory']).shape:
        distance = np.linalg.norm(np.array(filtered_response['trajectory']) - np.array(data['gold']['trajectory']), axis=1)
        de = np.mean(distance)
    else:
        de = 'inf'

    return {
        "de": de,
        "lateral_control_acc": 1.0
        if len(filtered_response['lateral_control']) / len(data['gold']['lateral_control']) > 0.8 and filtered_response['lateral_control'].lower() in data['gold']['lateral_control'].lower()
        else 0.0,
        "longitudinal_control_acc": 1.0
        if len(filtered_response['longitudinal_control']) / len(data['gold']['longitudinal_control']) > 0.8 and filtered_response['longitudinal_control'].lower() in data['gold']['longitudinal_control'].lower()
        else 0.0
    }

def estimate(scores, golds):
    lateral_group_names = [i['lateral_control'] for i in golds]
    longitudinal_group_names = [i['longitudinal_control'] for i in golds]

    def process_group_name2score(group_names, key):
        group_name2score, group_name2inf_count = defaultdict(list), defaultdict(int)
        cur_key = f'{key}_acc' if key != 'de' else 'de'
        for i_score, i_group in zip(scores, group_names):
            if i_score[cur_key] == 'inf':
                group_name2inf_count[i_group] += 1
            else:
                group_name2score[i_group].append(i_score[cur_key])

        for i in group_name2score.keys():
            group_name2score[i] = sum(group_name2score[i]) / max(len(group_name2score[i]), 1)
        return group_name2score, group_name2inf_count

    lateral_group_name2score, lateral_group_name2inf_count = process_group_name2score(lateral_group_names, key='lateral_control')
    longitudinal_group_name2score, longitudinal_group_name2inf_count = process_group_name2score(longitudinal_group_names, key='longitudinal_control')
    lateral_group_name2de, lateral_group_name2de_inf_count = process_group_name2score(lateral_group_names, key='de')
    longitudinal_group_name2de, longitudinal_group_name2de_inf_count = process_group_name2score(longitudinal_group_names, key='de')

    def process_acc(key):
        cur_scores = [i_score[f'{key}_acc'] for i_score in scores]
        return sum(cur_scores) / max(len(cur_scores), 1)
    des = [i_score['de'] for i_score in scores if i_score['de'] != 'inf']


    statistics = {
        'full': {
            'lateral_acc': process_acc('lateral_control'),
            'longitudinal_acc': process_acc('longitudinal_control'),
            'ade': sum(des) / max(len(des), 1),
            'lateral_worst_group_acc': min(lateral_group_name2score.items(), key=lambda x: x[1]) if len(lateral_group_name2score) > 0 else 'inf',
            'longitudinal_worst_group_acc': min(longitudinal_group_name2score.items(), key=lambda x: x[1]) if len(longitudinal_group_name2score) > 0 else 'inf',
            'lateral_worst_group_de': max(lateral_group_name2de.items(), key=lambda x: x[1]) if len(lateral_group_name2de) > 0 else 'inf',
            'longitudinal_worst_group_de': max(longitudinal_group_name2de.items(), key=lambda x: x[1]) if len(longitudinal_group_name2de) > 0 else 'inf',
            'lateral_harmonic_mean_acc': harmonic_mean(list(lateral_group_name2score.values())) if len(lateral_group_name2score) > 0 else 'inf',
            'longitudinal_harmonic_mean_acc': harmonic_mean(list(longitudinal_group_name2score.values())) if len(longitudinal_group_name2score) > 0 else 'inf',
            'lateral_harmonic_mean_de': harmonic_mean(list(lateral_group_name2de.values())) if len(lateral_group_name2de) > 0 else 'inf',
            'longitudinal_harmonic_mean_de': harmonic_mean(list(longitudinal_group_name2de.values())) if len(longitudinal_group_name2de) > 0 else 'inf',
            'f1_score': 0
        },
        **{f'{k} acc': v for k, v in lateral_group_name2score.items()},
        **{f'{k} acc': v for k, v in longitudinal_group_name2score.items()},
        **{f'{k} de': v for k, v in lateral_group_name2de.items()},
        **{f'{k} de': v for k, v in longitudinal_group_name2de.items()},

        **{f'{k} acc inf': v for k, v in lateral_group_name2inf_count.items()},
        **{f'{k} acc inf': v for k, v in longitudinal_group_name2inf_count.items()},
        **{f'{k} de inf': v for k, v in lateral_group_name2de_inf_count.items()},
        **{f'{k} de inf': v for k, v in longitudinal_group_name2de_inf_count.items()}
    }

    return statistics
