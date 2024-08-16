import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.loss import mesh_laplacian_smoothing


class LocalAffine(nn.Module):
    def __init__(self, num_points, batch_size, edges = None):
        '''
        self.A and self.b need to be updated: Ax = b
            self.A: torch.Size([1, 13106, 3, 3])
            self.b: torch.Size([1, 13106, 3, 1])
        '''
        super().__init__()
        # self.A and self.b need to be updated: Ax = b
        self.A = nn.Parameter(torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_points, 1, 1))
        self.b = nn.Parameter(torch.zeros(3).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(batch_size, num_points, 1, 1))
        self.edges = edges
        self.num_points = num_points


    def forward(self, x):
        '''
        calculate the stiffness of local affine transformation
        vertices should have shape of B * N * 3: Ax = b
        '''
        new_vertices = torch.matmul(self.A, x.unsqueeze(3)) + self.b
        new_vertices.squeeze_(3)

        affine_weight = torch.cat((self.A, self.b), dim = 3)
        w1 = torch.index_select(affine_weight, dim = 1, index = self.edges[:, 0])
        w2 = torch.index_select(affine_weight, dim = 1, index = self.edges[:, 1])
        stiffness = (w1 - w2) ** 2
        return new_vertices, stiffness
    


def nonrigidICP(source_mesh, target_mesh, config, gar_best, device = torch.device('cuda:0')):
    '''
    Deform source garment mesh to target human body
    The mesh should look at +z axis, the x define the width of mesh, and the y define the height of mesh
    '''
    
    source_mesh = source_mesh.to(device)
    target_mesh = target_mesh.to(device)

    source_vertices = source_mesh.verts_padded()
    target_vertices = target_mesh.verts_padded()

    K_neighbors = 8
    source_neighbors = []
    knn_search = knn_points(source_vertices, source_vertices, K = K_neighbors)
    for indice in knn_search.idx[0]:
        indice = indice.cpu().numpy()
        for i in range(1, K_neighbors):
            source_neighbors.append(sorted([indice[0], indice[i]]))

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
    laplacian_weight = config['laplacian_weight']
    if gar_best:
        num_iters = config['single_iter']   # 20
        stiffness_weight = np.array(config['best_alpha'])
    else:
        num_iters = config['all_iter']   # 150
        milestones = set(config['milestones'])
        stiffness_weight = np.array(config['alphas'])
        
    loop = tqdm(range(num_iters))
    
    stiff_idx = 0
    for _ in loop:
        new_verts, stiffness = local_affine_model(source_vertices)
        new_deform_mesh = source_mesh.update_padded(new_verts)

        knn_search = knn_points(new_verts, target_vertices)
        close_points = knn_gather(target_vertices, knn_search.idx)[:, :, 0]

        for _ in range(50):
            optimizer.zero_grad()

            vert_distance = (new_verts - close_points) ** 2
            vert_distance_mask = torch.sum(vert_distance, dim = 2) < 0.04**2
            vert_distance = vert_distance_mask.unsqueeze(2) * vert_distance

            vert_distance = vert_distance.view(1, -1)
            vert_sum = torch.sum(vert_distance)

            stiffness = stiffness.view(1, -1)
            stiffness_sum = torch.sum(stiffness) * stiffness_weight[stiff_idx]

            laplacian_loss = mesh_laplacian_smoothing(new_deform_mesh) * laplacian_weight
            
            loss = torch.sqrt(vert_sum + stiffness_sum) + laplacian_loss
            loss.backward()
            optimizer.step()
            
            new_verts, stiffness = local_affine_model(source_vertices)
            new_deform_mesh = source_mesh.update_padded(new_verts)
            
        if not gar_best: 
            if _ in milestones: 
                stiff_idx += 1

    new_deform_mesh = source_mesh.update_padded(new_verts)

    return new_deform_mesh

