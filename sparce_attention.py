import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from einops import rearrange, repeat
from timm.models.layers import DropPath
from einops import rearrange
from thop import profile, clever_format
import time
from common.utils import *
import random
from common.arguments import parse_args
import numpy as np
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'





def optimized_clustering_v2(x, cluster_num, k=2, token_mask=None):
    with torch.no_grad():
        B, N, C = x.shape

        # Reducción de dimensionalidad con una capa lineal o PCA para reducir el costo computacional
        reduction_layer = nn.Linear(C, 32).to(x.device)  # Reducción a 64 dimensiones
        x_reduced = reduction_layer(x)  # (B, N, 64)

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

# Funciones de atención normal
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

def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        b = torch.tril(torch.ones([n, n]))
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = min(n - 1, bandwidth - 1)
        b = torch.tril(torch.ones([n, n]), ctx)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
        y = torch.transpose(x, 0, 1)
        z = torch.zeros([n, n], dtype=torch.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = torch.eq(torch.fmod(q - k, stride), 0)
        c3 = torch.logical_and(c1, c2)
        b = c3.float()
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
    return torch.transpose(split_states(x, n), 0, 2, 1, 3)

def merge_heads(x):
    return merge_states(torch.transpose(x, 0, 2, 1, 3))

def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = x.size()
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return torch.reshape(x, new_x_shape)


    return torch.reshape(x, new_x_shape)

def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = x.size()
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return torch.reshape(x, new_x_shape)

def attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None):
    q = split_heads(q, heads)
    k = split_heads(k, heads)
    v = split_heads(v, heads)
    n_timesteps = k.size()[2]
    mask = get_attn_mask(n_timesteps, attn_mode, local_attn_ctx).float()
    w = torch.matmul(q, k.transpose(-2, -1))
    scale_amount = 1.0 / np.sqrt(q.size()[-1])
    w = w * scale_amount
    w = w * mask + -1e9 * (1 - mask)
    w = F.softmax(w, dim=-1)
    a = torch.matmul(w, v)
    a = merge_heads(a)
    return a

def blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None, blocksize=64, num_verts=None, vertsize=None):
    n_ctx = q.size()[1]
    
    # Imprimir las entradas
    #print(f"Input Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
    
    if attn_mode == 'strided':
        q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
        k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
        v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)

    # Imprimir después del strided_transpose
    #print(f"Strided Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
    
    n_state = q.size()[-1] // heads
    scale_amount = 1.0 / np.sqrt(n_state)
    
    # Imprimir el escalamiento
    #print(f"Scale amount: {scale_amount}")
    
    # Matriz de atención
    w = torch.matmul(q, k.transpose(-2, -1))
    
    # Imprimir matriz de atención antes de softmax
    #print(f"Attention Weights shape: {w.shape}, Attention Weights (before softmax): {w}")
    
    w = F.softmax(w * scale_amount, dim=-1)
    
    # Imprimir matriz de atención después de softmax
    #print(f"Attention Weights (after softmax): {w}")
    
    a = torch.matmul(w, v)#
    
    # Imprimir la salida de la atención
    #print(f"Attention Output shape: {a.shape}, Attention Output: {a}")
    
    if attn_mode == 'strided':
        n, t, embd = a.size()
        bT_ctx = n_ctx // local_attn_ctx
        a = torch.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
        a = torch.permute(a, (0, 2, 1, 3))
        a = torch.reshape(a, [n, t, embd])
    
    # Imprimir la salida final de la atención
    #print(f"Final Attention Output shape: {a.shape}, Final Attention Output: {a}")
    
    return a


class SparseAttention(nn.Module):
    def __init__(self, heads, attn_mode, local_attn_ctx=None, blocksize=32):
        super(SparseAttention, self).__init__()
        self.heads = heads
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.blocksize = blocksize

    def forward(self, q, k, v):
        return blocksparse_attention_impl(q, k, v, self.heads, self.attn_mode, self.local_attn_ctx,self.blocksize)


class SparseFinerAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sparsity=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        # Proyecciones para Q, K y V
        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sparsity = sparsity  # Factor de esparcidad para la atención

    def forward(self, x):
        B, N, C = x.shape
        
        # Cálculo de las proyecciones
        q = self.linear_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.linear_k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.linear_v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Cálculo de la atención
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Muestreo aleatorio de los top k valores
        num_keep = max(1, int(N * self.sparsity))  # Asegurar al menos 1
        topk_values, topk_indices = torch.topk(attn, num_keep, dim=-1)

        # Crear máscara de atención solo para los valores relevantes
        attn_mask = torch.zeros_like(attn).to(attn.device)
        attn_mask.scatter_(-1, topk_indices, topk_values)
        
        # Dropout de atención
        attn = self.attn_drop(attn_mask)

        # Aplicación de la atención a V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Proyección final
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=0,use_sparse=True,tipo='SparseFinerAttention'):
        super().__init__()

        self.norm1 = norm_layer(dim)
        if use_sparse:
            self.attn = SparseAttention(
                heads=num_heads, 
                local_attn_ctx = 1,
                attn_mode=tipo,
                blocksize = 1
            )
        else:
            if tipo == 'SparseFinerAttention':
                self.attn = SparseFinerAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            else: 
                self.attn = Attention(dim, num_heads=num_heads)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        if isinstance(self.attn, SparseAttention):
            q = k = v = self.norm1(x)
            attn_output = self.attn(q, k, v)
        else:
            attn_output = self.attn(self.norm1(x))
        
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
        
        
class Model(nn.Module):
    def __init__(self, args, use_sparse=True, tipo='all'):
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
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,use_sparse=use_sparse,tipo=tipo)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, depth=depth,use_sparse=use_sparse,tipo=tipo)
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
                index, idx_cluster = optimized_clustering_v2(x_knn, self.token_num, 2)
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

def benchmark_attention(model, input_data):
    # Calcular FLOPs y tiempo de ejecución
    macs, params = profile(model, inputs=(input_data,))  # Calcula FLOPs
    macs, params = clever_format([macs, params], "%.3f")  # Formatear los resultados

    # Medir el tiempo de ejecución
    start_time = time.time()
    with torch.no_grad():
        output = model(input_data)
    elapsed_time = time.time() - start_time

    return macs, params, elapsed_time

# Ejemplo de uso
if __name__ == "__main__":
    seed = 1126

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Preparar datos de entrada
    input_data = torch.randn(2, 243, 17, 2)  # Cambia el tamaño del batch según sea necesario

    # Modelo con Attention Normal
    normal_model = Model(args, use_sparse=False)  # Aquí se utiliza la atención normal
    Load_model(args, normal_model)
    normal_macs, normal_params, normal_time = benchmark_attention(normal_model, input_data)

    print(f"----------------------------------------Normal Attention - FLOPs: {normal_macs}, Params: {normal_params}, Time: {normal_time:.4f} seconds")

    # Modelo con Efficient Attention
    efficient_model = Model(args, use_sparse=True,tipo='all') # Aquí se utiliza la atención eficiente
    Load_model(args, efficient_model)
    efficient_macs, efficient_params, efficient_time = benchmark_attention(efficient_model, input_data)

    print(f"----------------------------------------Efficient Attention ALL - FLOPs: {efficient_macs}, Params: {efficient_params}, Time: {efficient_time:.4f} seconds")

    # Modelo con Efficient Attention
    efficient_model = Model(args, use_sparse=True,tipo='local') # Aquí se utiliza la atención eficiente
    Load_model(args, efficient_model)
    efficient_macs, efficient_params, efficient_time = benchmark_attention(efficient_model, input_data)

    print(f"----------------------------------------Efficient Attention LOCAL - FLOPs: {efficient_macs}, Params: {efficient_params}, Time: {efficient_time:.4f} seconds")
    # Modelo con Efficient Attention
    efficient_model = Model(args, use_sparse=True,tipo='strided') # Aquí se utiliza la atención eficiente
    Load_model(args, efficient_model)
    efficient_macs, efficient_params, efficient_time = benchmark_attention(efficient_model, input_data)

    print(f"----------------------------------------Efficient Attention STRIDE - FLOPs: {efficient_macs}, Params: {efficient_params}, Time: {efficient_time:.4f} seconds")

    
    # Modelo con Attention Normal
    normal_model = Model(args, use_sparse=False,tipo= 'SparseFinerAttention')  # Aquí se utiliza la atención normal
    Load_model(args, normal_model)
    normal_macs, normal_params, normal_time = benchmark_attention(normal_model, input_data)

    print(f"----------------------------------------SparseFinerAttention Attention - FLOPs: {normal_macs}, Params: {normal_params}, Time: {normal_time:.4f} seconds")



    