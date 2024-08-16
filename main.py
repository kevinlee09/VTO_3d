import os
import json
import torch
import argparse
from utils.nonrigid_icp import nonrigidICP
from utils.post_process import solve_collision
from pytorch3d.io import load_objs_as_meshes, save_obj


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="3D Virtual Try-on!")
    parser.add_argument('--body', default="./data/body/female_hres_1_-1.obj", type=str)
    parser.add_argument('--garment', default="./data/garment/garment_test.obj", type=str)
    parser.add_argument('--out_path', default="./results", type=str)      

    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    source_path = args.garment
    target_path = args.body
    result_path = args.out_path
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
    body_shape = target_path.replace(".obj", "")[-4:]

    """
    garment registration on human body
        gar_best: the best alpha setting for all garments
    """
    gar_best = True

    source_mesh = load_objs_as_meshes([source_path], device = device)
    target_mesh = load_objs_as_meshes([target_path], device = device)

    nricp_params = json.load(open('config/nricp.json'))
    post_process_params = json.load(open('config/post_process.json'))
    
    alphas = list(nricp_params['best_alpha'])
    new_mesh = nonrigidICP(source_mesh, target_mesh, nricp_params, gar_best, device)
    
    save_path = os.path.join(result_path, f"nricp_result_{body_shape}.obj")
    save_obj(save_path, new_mesh.verts_padded()[0], new_mesh.faces_padded()[0])   ### nricp results
     
     
    """ 
    Post process 
    """
    K_neighbors = post_process_params["K_neighbors"]
    eps = post_process_params["eps"]
    iter= post_process_params['iter']
    new_mesh = solve_collision(save_path, target_path, K_neighbors, eps, iter, device)
    save_obj(os.path.join(result_path, f"postprocess_result_{body_shape}.obj"), new_mesh.verts_padded()[0], new_mesh.faces_padded()[0])
    


