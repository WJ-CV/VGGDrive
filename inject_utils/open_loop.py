import os
import pickle
import json
import torch
import numpy as np
import ast
import argparse

def compute_ade(trajs, gt_trajs):
    trajs = torch.tensor(trajs, dtype=torch.float32)
    gt_trajs = torch.tensor(gt_trajs, dtype=torch.float32)
    ade = torch.norm(trajs - gt_trajs, dim=1).mean().item()
    return ade

def compute_L2(pkl_data):
    ades_1s = []
    ades_2s = []
    ades_3s = []
    for item in pkl_data['predictions']:
        try:
            pred = np.array(item['pre_traj'])
            gt = np.array(ast.literal_eval(item['messages'][1]['content']))
            # pred = np.cumsum(pred, axis=0)   
            # gt = np.cumsum(gt[:, :2], axis=0) 

            ade_1s = compute_ade(pred[:2], gt[:2])
            ades_1s.append(ade_1s)
            ade_2s = compute_ade(pred[:4], gt[:4])
            ades_2s.append(ade_2s)
            ade_3s = compute_ade(pred, gt)
            ades_3s.append(ade_3s)
        except Exception as e:
            print(f"Skipping ID {item['id']} due to error: {e}")

    avg_ade_1s = np.mean(ades_1s)
    avg_ade_2s = np.mean(ades_2s)
    avg_ade_3s = np.mean(ades_3s)

    return len(ades_1s), avg_ade_1s, avg_ade_2s, avg_ade_3s

def compute_L2_4s(pkl_data):
    ades_1s = []
    ades_2s = []
    ades_3s = []
    ades_4s = []
    for item in pkl_data['predictions']:
        try:
            pred = np.array(item['pre_traj'])
            gt = np.array(ast.literal_eval(item['messages'][1]['content']))
            # pred = np.cumsum(pred, axis=0)   
            # gt = np.cumsum(gt[:, :2], axis=0) 

            ade_1s = compute_ade(pred[:2], gt[:2])
            ades_1s.append(ade_1s)
            ade_2s = compute_ade(pred[:4], gt[:4])
            ades_2s.append(ade_2s)
            ade_3s = compute_ade(pred[:6], gt[:6])
            ades_3s.append(ade_3s)
            ade_4s = compute_ade(pred, gt)
            ades_4s.append(ade_4s)
        except Exception as e:
            print(f"Skipping ID {item['id']} due to error: {e}")

    avg_ade_1s = np.mean(ades_1s)
    avg_ade_2s = np.mean(ades_2s)
    avg_ade_3s = np.mean(ades_3s)
    avg_ade_4s = np.mean(ades_4s)

    return len(ades_1s), avg_ade_1s, avg_ade_2s, avg_ade_3s, avg_ade_4s

def merge_data_ADE_3s(output_path, replace_name):
    save_path = output_path.replace('results.pkl', replace_name)
    if not os.path.exists(save_path):
        test_data_json = './cache/navsim_4s_test.json'
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
        with open(test_data_json, 'r') as f:
            json_data = json.load(f)

        id_to_messages = {item['id']: item['messages'] for item in json_data}

        for item in data['predictions']:
            if len(item['pre_traj']) > 6:
                item['pre_traj'] = item['pre_traj'][:6]
            item_id = item['id']
            if item_id in id_to_messages:
                item['messages'] = id_to_messages[item_id]
            else:
                print(f"[Warning] ID '{item_id}' not found in JSON file. Skipping.")

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[Done] Merged data saved to: {save_path}")

    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    num, avg_ade_1s, avg_ade_2s, avg_ade_3s = compute_L2(data)
    avg_ade = (avg_ade_1s+avg_ade_2s+avg_ade_3s) / 3
    print(f"samples: {num}, ADE: 1s-{avg_ade_1s:.4f} meters, 2s-{avg_ade_2s:.4f} meters, 3s-{avg_ade_3s:.4f} meters, avg-{avg_ade:.4f}")


def merge_data_ADE_4s(output_path, replace_name):
    save_path = output_path.replace('results.pkl', replace_name)
    if not os.path.exists(save_path):
        test_data_json = 'navsim_dataset/cache/navsim_4s_test.json'
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
        with open(test_data_json, 'r') as f:
            json_data = json.load(f)

        id_to_messages = {item['id']: item['messages'] for item in json_data}

        for item in data['predictions']:
            if len(item['pre_traj']) > 8:
                item['pre_traj'] = item['pre_traj'][:8]

            item_id = item['id']
            if item_id in id_to_messages:
                item['messages'] = id_to_messages[item_id]
            else:
                print(f"[Warning] ID '{item_id}' not found in JSON file. Skipping.")

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[Done] Merged data saved to: {save_path}")

    with open(save_path, 'rb') as f:
        data = json.load(f)
    num, avg_ade_1s, avg_ade_2s, avg_ade_3s, avg_ade_4s = compute_L2_4s(data)
    avg_ade = (avg_ade_1s+avg_ade_2s+avg_ade_3s+avg_ade_4s) / 4
    print(f"samples: {num}, ADE: 1s-{avg_ade_1s:.4f} meters, 2s-{avg_ade_2s:.4f} meters, 3s-{avg_ade_3s:.4f} meters, , 4s-{avg_ade_4s:.4f} meters, avg-{avg_ade:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True, help='Path to output .pkl file')
    parser.add_argument('--filename', type=str, required=True, help='Special token filename')
    args = parser.parse_args()

    merge_data_ADE_4s(args.output_path, args.filename)
