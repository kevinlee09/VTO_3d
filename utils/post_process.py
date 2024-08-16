import torch
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from pytorch3d.ops import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.io import load_objs_as_meshes, load_obj
from utils.local_affine import LocalAffine


def compute_vertex_normals(vertices, faces):
    # Vertex normals weighted by triangle areas:
    normals = np.zeros(vertices.shape, dtype=vertices.dtype)
    triangles = vertices[faces]

    e1 = triangles[::, 0] - triangles[::, 1]    
    e2 = triangles[::, 2] - triangles[::, 1]
    n = np.cross(e2, e1) 

    np.add.at(normals, faces[:,0], n)
    np.add.at(normals, faces[:,1], n)
    np.add.at(normals, faces[:,2], n)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / norms


def find_collisions(vc, vb, nb, eps, flag):
    # For each vertex of the cloth, find the closest vertices in the body's surface
    search_nearest = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(vb)
    _, indices = search_nearest.kneighbors(vc)
    closest_vertices = [i[0] for i in list(indices)]
    
    vb = vb[closest_vertices] 
    nb = nb[closest_vertices] 

    # Test penetrations
    penetrations = np.sum(nb*(vc - vb), axis=1) - eps
    penetrations = np.minimum(penetrations, 0)

    # find all penetrations locations
    pene_locs = np.argwhere(penetrations < 0).flatten()
    if flag: 
        penetrations[np.where(penetrations < 0)] = -0.005
    
    return nb, penetrations, pene_locs
    


def affine_solve(gar_mesh, vc_new, source_neighbors, device = torch.device('cuda:0')):
    '''
    Deform source garment mesh to target human body
    The mesh should look at +z axis, the x define the width of mesh, and the y define the height of mesh
    '''
    source_vertices = gar_mesh.verts_padded()
    
    vc_deformed = torch.from_numpy(vc_new).unsqueeze(0).float().to(device)

    # using k neighbors to substitute the original edges
    source_neighbors = torch.from_numpy(np.array(source_neighbors)).to(device)   
    
    '''
    Local affine model: used to evaluate stiffness term
        arg1: num of points;
        arg2: size of batch
        arg3: conectivity of mesh (edges of neighbors)
    '''
    local_affine_model = LocalAffine(source_vertices.shape[1], source_vertices.shape[0], source_neighbors).to(device)
    optimizer = torch.optim.AdamW([{'params': local_affine_model.parameters()}], lr=1e-4, amsgrad=True)
        
    '''
    Get best train parameters configuration
    '''
    smooth_weight = 10
    num_iters = 20   # 20
    stiffness_weight = np.array([100])
    loop = tqdm(range(num_iters))


    stiff_idx = 0
    for _ in loop:
        # print("iter: {}, with alpha = {}".format(iter, stiffness_weight[stiff_idx]))
        
        new_verts, stiffness = local_affine_model(source_vertices)
        new_deform_mesh = gar_mesh.update_padded(new_verts)

        for i in range(10):
            optimizer.zero_grad()

            vert_distance = (new_verts - vc_deformed) ** 2
            vert_distance_mask = torch.sum(vert_distance, dim = 2) <  0.04**2
            vert_distance = vert_distance_mask.unsqueeze(2) * vert_distance
            
            vert_distance = vert_distance.view(1, -1)
            vert_sum = torch.sum(vert_distance)

            stiffness_term = stiffness.view(1, -1)
            stiffness_sum = torch.sum(stiffness_term) * stiffness_weight[stiff_idx]

            smooth_loss = mesh_laplacian_smoothing(new_deform_mesh) * smooth_weight

            loss = torch.sqrt(50 * vert_sum + stiffness_sum) + smooth_loss
            loss.backward()

            optimizer.step()
            
            new_verts, stiffness = local_affine_model(source_vertices)
            new_deform_mesh = gar_mesh.update_padded(new_verts) 


    new_deform_mesh = gar_mesh.update_padded(new_verts)

    return new_deform_mesh



def solve_collision(cloth_path, body_path, K_neighbors, eps, iter_num, device):
    vc_tensor, fc_tensor, _ = load_obj(cloth_path)
    vb_tensor, fb_tensor, _ = load_obj(body_path)

    vc = vc_tensor.detach().cpu().numpy()
    fc = fc_tensor.verts_idx.detach().cpu().numpy()
    vb = vb_tensor.detach().cpu().numpy()
    fb = fb_tensor.verts_idx.detach().cpu().numpy()
    
    nb = compute_vertex_normals(vb, fb)
    print("#verts of garment: ", vc.shape)
    
    
    '''
    Reconnect the mesh (find nearest K neighbors as its connected points)
    '''
    source_mesh = load_objs_as_meshes([cloth_path], device = device) 
    source_mesh = source_mesh.to(device)
    source_vertices = source_mesh.verts_padded()

    source_neighbors = []
    knn_search = knn_points(source_vertices, source_vertices, K = K_neighbors)
    for indice in knn_search.idx[0]:
        indice = indice.cpu().numpy()
        for i in range(1, K_neighbors):
            source_neighbors.append(sorted([indice[0], indice[i]]))
    

    gar_mesh = source_mesh
    i = 0
    for iter in range(iter_num):
        print("\nIter: ", iter)

        if iter != 0:
            vc_tensor = gar_mesh.verts_padded()
            vc = torch.squeeze(vc_tensor).detach().cpu().numpy()
        
        _, _, gar_locs = find_collisions(vc, vb, nb, eps, False)
        if len(gar_locs) < 30:
            print("inside verts num: ", len(gar_locs))
            new_nb, penetrations, gar_locs = find_collisions(vc, vb, nb, eps * 2, False)
        
            corrective_offset = -np.multiply(new_nb, penetrations[:, np.newaxis])
            vc_fixed = vc + corrective_offset
            
            new_verts = torch.from_numpy(vc_fixed).unsqueeze(0).float().to(device)
            gar_mesh = gar_mesh.update_padded(new_verts)
            break
        else:
            print("inside verts num: ", len(gar_locs))
            new_nb, penetrations, gar_locs = find_collisions(vc, vb, nb, eps * 2, True)

            corrective_offset = -np.multiply(new_nb, penetrations[:, np.newaxis])
            vc_new = vc + corrective_offset
            
            gar_mesh = affine_solve(gar_mesh, vc_new, source_neighbors, device)
            gar_mesh = gar_mesh.detach()
        
        i += 1
    
    
    for iter in range(10):
        print("\nIter: ", iter + i + 1)

        vc_tensor = gar_mesh.verts_padded()
        vc = torch.squeeze(vc_tensor).detach().cpu().numpy()
            
        _, _, gar_locs = find_collisions(vc, vb, nb, eps, False)
        if len(gar_locs) == 0:
            print("no collision now!")
            return gar_mesh
        else: 
            print("inside verts num: ", len(gar_locs))
            new_nb, penetrations, gar_locs = find_collisions(vc, vb, nb, eps, False)
        
            corrective_offset = -np.multiply(new_nb, penetrations[:, np.newaxis])
            vc_fixed = vc + corrective_offset
            
            new_verts = torch.from_numpy(vc_fixed).unsqueeze(0).float().to(device)
            gar_mesh = gar_mesh.update_padded(new_verts)
    
    return gar_mesh