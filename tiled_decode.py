import torch, math
from einops import rearrange

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def remove_overlap(tile, n_slices, overlap, i, j, h, w):
    overlap_ = overlap * 8
    sh = 0 if i == 0 else overlap_ // 2
    eh = h * 8 - overlap_ // 2 if i != n_slices - 1 else h*8 
    sw = 0 if j == 0 else overlap_ // 2
    ew = w*8 - overlap_ // 2 if j != n_slices - 1 else w*8 
    tile = tile[:, :, sh:eh, sw:ew]
    return tile

def tiled_vae_decoding(model, x, window_size, overlap, sync_gn=False):
    """
    Args:
        x: latent for decoding
        window_size: (h, w) of tile shape 
        overlap: overlapped length between tiles
        sync_gn: sync GN between tiles
    """
    assert(overlap % 2 == 0)
    B, C, H, W = x.shape
    h, w = window_size, window_size

    if overlap == 0:
        # no overlapped tiling
        if sync_gn:
            x = window_partition(x.permute(0,2,3,1), window_size=window_size).permute(0,3,1,2)
            tiles = [x_.unsqueeze(0) for x_ in x]
            tiles = model.decode_first_stage_tiles(tiles)
            x = torch.cat(tiles, dim=0)
        else:
            x = window_partition(x.permute(0,2,3,1), window_size=window_size).permute(0,3,1,2)
            x = model.decode_first_stage(x)
        return window_reverse(x.permute(0,2,3,1), window_size*8, H*8,W*8).permute(0,3,1,2)
    
    # overlapped tiling
    stride = h-overlap
    n_slices= math.ceil((H - h)/(h-overlap)) + 1

    if sync_gn:
        tiles = []
        for i in range(n_slices):
            for j in range(n_slices):
                tiles.append(x[:, :, i*stride:i*stride+h, j*stride:j*stride+h])
        tiles = model.decode_first_stage_tiles(tiles)
        
        outs = []
        for i in range(n_slices):
            for j in range(n_slices):
                tile = remove_overlap(tiles[i*n_slices+j], n_slices, overlap, i, j, h, w)
                outs.append(tile)
    else:
        outs = []
        for i in range(n_slices):
            for j in range(n_slices):
                out = x[:, :, i*stride:i*stride+h, j*stride:j*stride+h]
                out = model.decode_first_stage(out)
                tile = remove_overlap(out, n_slices, overlap, i, j, h, w)
                outs.append(tile)
    # merge tiles
    rows=[]
    for i in range(n_slices):
        rows.append(torch.cat(outs[i*n_slices:(i+1)*n_slices], dim=3))
    outs = torch.cat(rows, dim=2)
    return outs

def make_conv(inchs, outchs, tiled=False, *args, **kwargs):
    if tiled:
        return TiledConv2d(inchs, outchs, *args, **kwargs)
    else:
        return torch.nn.Conv2d(inchs, outchs, *args, **kwargs)

class TiledConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, slices):
        if isinstance(slices, list):
            return [self._conv_forward(slice, self.weight, self.bias) for slice in slices]
        else:
            return self._conv_forward(slices, self.weight, self.bias)

def make_norm(in_channels, num_groups=32, sync_gn=False):
    if sync_gn:
        return TileSyncGN(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class TileSyncGN(torch.nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, slices):
        b = slices[0].shape[0]
        self.dtype = slices[0].dtype
        self.device = slices[0].device
        mean = torch.zeros((b, self.num_groups, 1), dtype=self.dtype, device=self.device)
        var = torch.zeros((b, self.num_groups, 1), dtype=self.dtype, device=self.device)
        num_elements = 0
        slice_shapes = list()
        ns=len(slices)
        slices2=[]
        for slice in slices:
            b, c, h, w = slice.shape
            slice_shapes.append((b, c, h, w))
            slice = rearrange(slice, 'b (g c) h w -> b g (c h w)', g=self.num_groups)
            num_elements = num_elements + slice.shape[-1]
            slices2.append(slice)
        slices = slices2

        for slice in slices:
            slice.to(self.device)
            try:
                mean = mean + slice.mean(-1, keepdim=True) * float(slice.shape[-1] / num_elements)
            except:
                import pdb;pdb.set_trace()
            slice.to('cpu')

        for slice in slices:
            slice.to(self.device)
            var = var + (
                    ((slice - mean) ** 2) * (slice.shape[-1] / (slice.shape[-1] - 1))
            ).mean(-1, keepdim=True) * float(slice.shape[-1] / num_elements)
            slice.to('cpu')
        
        slices2=[]
        for i, slice in enumerate(slices):
            slice.to(self.device)
            b, c, h, w = slice_shapes[i]
            slice = (slice - mean) / (var + self.eps).sqrt()
            slice = rearrange(slice, 'b g (c h w) -> b (g c) h w', h=h, w=w) 
            slice = slice * self.weight.unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1)
            slice.to('cpu')
            slices2.append(slice)
        slices = slices2
        slices = [slice.to(self.device) for slice in slices]
        return slices

def nonlinearity(x):
    # swish
    if isinstance(x, list):
        return [x_*torch.sigmoid(x_) for x_ in x]
    else:
        return x*torch.sigmoid(x)
