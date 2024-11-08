import sys
import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from einops import rearrange, repeat
import numpy as np
import torch.nn.functional as F


def optimized_clustering_v2(x, cluster_num, k=2, token_mask=None):
    with torch.no_grad():
        B, N, C = x.shape

        x_reduced = x

        # Calcular la matriz de distancias con cdist (más eficiente con dimensionalidad reducida)
        dist_matrix = torch.cdist(x_reduced, x_reduced) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # Seleccionar los k vecinos más cercanos usando torch.topk (más eficiente que cdist denso)
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        # Calcular la densidad basada en las distancias
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            density = density * token_mask

        # Crear máscara para comparar densidades
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)

        # Calcular el valor máximo en la matriz de distancias
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]

        # Calcular la distancia mínima a los padres usando la máscara
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # Calcular las puntuaciones basadas en distancia y densidad
        score = dist * density

        # Seleccionar los clusters con mayores puntuaciones
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # Recalcular la matriz de distancias solo para los clusters seleccionados
        dist_matrix = torch.zeros(B, cluster_num, N, device=x.device)
        for i in range(B):
            dist_matrix[i] = torch.cdist(x_reduced[i, index_down[i]], x_reduced[i]) / (C ** 0.5)

        # Encontrar el cluster más cercano para cada punto
        idx_cluster = dist_matrix.argmin(dim=1)

        # Ajustar los índices del cluster
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down, idx_cluster

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points


def cluster_dpc_knn(x, cluster_num, k, token_mask=None):
    with torch.no_grad():
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            density = density * token_mask

        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        score = dist * density
        
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down, idx_cluster


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., length=27):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2, x_3):
        B, N, C = x_1.shape
        B, N_1, C = x_3.shape

        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_3).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_attn_mask(n, attn_mode, local_attn_ctx=None, sparse_stride=2):
    if attn_mode == 'all':
        b = torch.ones([n, n])
    elif attn_mode == 'local':
        b = torch.zeros((n, n), dtype=torch.float32)
        idx = torch.arange(n)
        # Crear el índice de ventanas desplazadas por el contexto local
        for i in range(-local_attn_ctx, local_attn_ctx + 1):
            b += torch.diag(torch.ones(n - abs(i)), i)
    elif attn_mode == 'strided':
        b = torch.zeros((n, n), dtype=torch.float32)
        for i in range(0, n, local_attn_ctx):
            b[i, :] = 1.0  # Permitir que el frame i atienda a todos los frames
            b[:, i] = 1.0  # Permitir que todos los frames atiendan al frame i
 
    elif attn_mode == 'dense_sparse':
        # Crear la matriz de atención
        b = torch.zeros((n, n), dtype=torch.float32)

        # Agregar atención densa usando ventana local
        for i in range(n):
            # Ventana local
            start = max(0, i - local_attn_ctx)
            end = min(n, i + local_attn_ctx + 1)
            b[i, start:end] = 1.0  # Atención densa

        # Agregar atención escasa
        if sparse_stride is not None:
            for i in range(0, n, sparse_stride):
                b[i, :] = 1.0  # Permitir que el frame i atienda a todos los frames
                b[:, i] = 1.0  # Permitir que todos los frames atiendan al frame i

    else:
        raise ValueError('Not yet implemented')
    
    b = torch.reshape(b, [1, 1, n, n])
    return b


def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    bT_ctx = n_ctx // local_attn_ctx
    
    # Imprimir el cálculo de bT_ctx
    #print(f"bT_ctx: {bT_ctx}, blocksize: {blocksize}")
    
    assert bT_ctx % blocksize == 0, f'{bT_ctx}, {blocksize}'
    
    n, t, embd = x.size()
    x = torch.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
    
    # Imprimir después del reshape
    #print(f"After reshape, X shape: {x.shape}")
    
    x = torch.permute(x, (0, 2, 1, 3))
    
    # Imprimir después de permute
    #print(f"After permute, X shape: {x.shape}")
    
    x = torch.reshape(x, [n, t, embd])
    
    # Imprimir la forma final
    #print(f"Final Transposed X shape: {x.shape}")
    
    return x


def split_heads(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = x.size()
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + (n, m // n)
    x = torch.reshape(x, new_x_shape)
    # Cambia esta línea
    return torch.permute(x, (0, 2, 1, 3))  # Usamos permute en lugar de transpose


def merge_heads(x):
    # Asumiendo que x tiene la forma [batch_size, heads, seq_length, head_dim]
    # Cambia la forma de x para que tenga la forma [batch_size, seq_length, heads * head_dim]
    # Asegúrate de que las dimensiones a transponer son correctas según la forma de tu tensor
    x = torch.transpose(x, 1, 2)  # Cambia la dimensión de los heads y seq_length
    # Luego, combina los heads
    return x.reshape(x.size(0), x.size(1), -1) 

def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = x.size()
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + (n, m // n)
    return torch.reshape(x, new_x_shape)

def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = x.size()
    new_x_shape = x_shape[:-2] + (np.prod(x_shape[-2:]))
    return torch.reshape(x, new_x_shape)

def attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None):
    q = split_heads(q, heads)
    k = split_heads(k, heads)
    v = split_heads(v, heads)
    n_timesteps = k.size()[2]
    mask = get_attn_mask(n_timesteps, attn_mode, local_attn_ctx).float()
    
    # Mover el mask a la misma GPU que los tensores q y k
    mask = mask.to(q.device)

    w = torch.matmul(q, k.transpose(-2, -1))
    scale_amount = 1.0 / np.sqrt(q.size()[-1])
    w = w * scale_amount
    w = w * mask + -1e9 * (1 - mask)
    w = F.softmax(w, dim=-1)
    a = torch.matmul(w, v)
    a = merge_heads(a)
    return a


# def blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None, blocksize=64, num_verts=None, vertsize=None):
#     n_ctx = q.size()[1]
    
#     # Imprimir las entradas
#     #print(f"Input Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
    
#     if attn_mode == 'strided':
#         q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
#         k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
#         v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)

#     # Imprimir después del strided_transpose
#     #print(f"Strided Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
    
#     n_state = q.size()[-1] // heads
#     scale_amount = 1.0 / np.sqrt(n_state)
    
#     # Imprimir el escalamiento
#     #print(f"Scale amount: {scale_amount}")
    
#     # Matriz de atención
#     w = torch.matmul(q, k.transpose(-2, -1))
    
#     # Imprimir matriz de atención antes de softmax
#     #print(f"Attention Weights shape: {w.shape}, Attention Weights (before softmax): {w}")
    
#     w = F.softmax(w * scale_amount, dim=-1)
    
#     # Imprimir matriz de atención después de softmax
#     #print(f"Attention Weights (after softmax): {w}")
    
#     a = torch.matmul(w, v)#
    
#     # Imprimir la salida de la atención
#     #print(f"Attention Output shape: {a.shape}, Attention Output: {a}")
    
#     if attn_mode == 'strided':
#         n, t, embd = a.size()
#         bT_ctx = n_ctx // local_attn_ctx
#         a = torch.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
#         a = torch.permute(a, (0, 2, 1, 3))
#         a = torch.reshape(a, [n, t, embd])
    
#     # Imprimir la salida final de la atención
#     #print(f"Final Attention Output shape: {a.shape}, Final Attention Output: {a}")
    
#     return a


class SparseAttention(nn.Module):
    def __init__(self, heads, attn_mode, local_attn_ctx=None, blocksize=17):
        super(SparseAttention, self).__init__()
        self.heads = heads
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.blocksize = blocksize

    def forward(self, q, k, v):
        return attention_impl(q, k, v, self.heads, self.attn_mode, self.local_attn_ctx)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=0, type = 'spatial'):
        super().__init__()

        self.type = type 
        self.norm1 = norm_layer(dim)
        self.attn = SparseAttention(
                heads=num_heads, 
                local_attn_ctx = 2,
                attn_mode='strided', 
            )

        self.attn2 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        if self.type  == 'spatial': 
            attn_output = self.attn2(self.norm1(x))
        elif self.type  == 'temporal': 
            q = k = v = self.norm1(x)
            attn_output = self.attn(q, k, v)
        
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        depth = 8
        embed_dim = args.channel
        mlp_hidden_dim = args.channel * 2

        self.center = (args.frames - 1) // 2

        self.recover_num = args.frames
        self.token_num = args.token_num
        self.layer_index = args.layer_index

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pos_embed_token = nn.Parameter(torch.zeros(1, self.token_num, embed_dim))

        drop_path_rate = 0.1
        drop_rate = 0.
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None

        num_heads = 8
        num_joints = args.n_joints

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.Spatial_patch_to_embedding = nn.Linear(2, embed_dim)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, args.frames, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, type = 'spatial')
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, depth=depth, type = 'temporal')
            for i in range(depth)])

        self.x_token = nn.Parameter(torch.zeros(1, self.recover_num, embed_dim))

        self.cross_attention = Cross_Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate)

        self.Spatial_norm = norm_layer(embed_dim)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , 3),
        )

    def forward(self, x):
        b, f, n, c = x.shape

        x = rearrange(x, 'b f n c  -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)

        x = rearrange(x, '(b f) n c -> (b n) f c', f=f)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)

        x = rearrange(x, '(b n) f c -> b f n c', n=n)
        for i in range(1, self.block_depth):
            ##-----------------Clusteing-----------------##
            if i == self.layer_index:
                x_knn = rearrange(x, 'b f n c -> b (f c) n')
                x_knn = self.pool(x_knn)
                x_knn = rearrange(x_knn, 'b (f c) 1 -> b f c', f=f)

                # Aplicar clustering para seleccionar los tokens más relevantes
                index, idx_cluster = cluster_dpc_knn(x_knn, self.token_num, 2)
                index, _ = torch.sort(index)  # Selección de los tokens representativos

                # Filtrar los tokens seleccionados
                batch_ind = torch.arange(b, device=x.device).unsqueeze(-1)
                x = x[batch_ind, index]  # Solo pasar los tokens seleccionados

                x = rearrange(x, 'b f n c -> (b n) f c')
                x += self.pos_embed_token
                x = rearrange(x, '(b n) f c -> b f n c', n=n)

            x = rearrange(x, 'b f n c -> (b f) n c')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n c -> (b n) f c', b=b)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f c -> b f n c', n=n)

        x = rearrange(x, 'b f n c -> (b n) f c')
        x_token = repeat(self.x_token, '() f c -> b f c', b = b*n)
        x = x_token + self.cross_attention(x_token, x, x)
        x = rearrange(x, '(b n) f c -> b f n c', n=n)

        x = self.head(x)

        x = x.view(b, -1, n, 3)

        return x
import torch.profiler
if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser().parse_args()
    args.layers, args.channel, args.d_hid, args.frames = 8, 512, 1024, 243
    args.n_joints, args.out_joints = 17, 17
    args.token_num = 81
    args.layer_index = 3

    input_2d = torch.rand(1, args.frames, 17, 2)

    with torch.no_grad():
        model = Model(args)
        model.eval()

        model_params = 0
        for parameter in model.parameters():
            model_params += parameter.numel()
        print('INFO: Trainable parameter count:', model_params/ 1000000)

        with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],  # CUDA si estás usando GPU
        ) as prof:
            output = model(input_2d)

    print(prof.key_averages().table(sort_by="cpu_time_total"))

    from thop import profile
    from thop import clever_format
    from fvcore.nn import FlopCountAnalysis
    import time
    import matplotlib.pyplot as plt
    
    flops = FlopCountAnalysis(model, input_2d)
    print("Total FLOPs: ", flops.total())


    block = model.STEblocks[3]
    attention = model.STEblocks[3].attn
    x = rearrange(input_2d, 'b f n c  -> (b f) n c')
    x = model.Spatial_patch_to_embedding(x)
    x += model.Spatial_pos_embed
    x = model.pos_drop(x)
    q = k = v = block.norm1(x)
    _ = attention(q, k, v)  # Asumiendo que attention_impl es tu función



    def calculate_flops_for_mask(mask, d):
        # mask tiene dimensiones [n, n] y cada entrada es 0 o 1
        # d es la dimensión de cada vector de características por cabeza
        active_elements = mask.sum().item()  # Cuenta cuántos 1s hay en la máscara
        flops_per_element = 2 * d - 1  # Multiplicaciones y sumas por elemento en el producto punto
        total_flops = flops_per_element * active_elements
        return total_flops


    n_timesteps = k.size()[2]
    mask = get_attn_mask(n_timesteps, 'dense_sparse', 2).float()
    mask2 = get_attn_mask(n_timesteps, 'local', 2).float()
    mask3 = get_attn_mask(n_timesteps, 'all', 2).float()
    mask4 = get_attn_mask(n_timesteps, 'strided', 2).float()
    d = 512  # Dimensiones por cabeza
    flops = calculate_flops_for_mask(mask, d)
    flops2 = calculate_flops_for_mask(mask2, d)
    flops3 = calculate_flops_for_mask(mask3, d)
    flops4 = calculate_flops_for_mask(mask4, d)
    print(f"Total FLOPs for dense_sparse attention with context 3: {flops}")
    print(f"Total FLOPs for local attention with context 3: {flops2}")
    print(f"Total FLOPs for all attention with context 3: {flops3}")
    print(f"Total FLOPs for strided attention with context 3: {flops4}")

    modes = ['local', 'strided', 'all']
    for mode in modes:
            mask = get_attn_mask(n_timesteps, mode, 5).float()
            mask = mask.squeeze()  # Reduce las dimensiones adicionales
            plt.figure(figsize=(8, 6))
            plt.imshow(mask.cpu().numpy(), cmap='viridis', aspect='auto')
            plt.title(f'Attention Mask for Mode: {mode}')
            plt.colorbar()
            plt.xlabel('Key Positions')
            plt.ylabel('Query Positions')
            plt.show()







