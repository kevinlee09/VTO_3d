import torch
import torch.nn as nn

class LocalAffine(nn.Module):
    def __init__(self, num_points, batch_size, edges = None):
        '''
        self.A and self.b need to be updated: Ax = b
            self.A: torch.Size([1, 13106, 3, 3])
            self.b: torch.Size([1, 13106, 3, 1])
        '''
        super(LocalAffine, self).__init__()
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
