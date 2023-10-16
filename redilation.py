
from torch import nn

def make_dilate_module(module, dilate):
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.Conv2d) and (submodule.kernel_size == 3 or submodule.kernel_size == (3,3)):
            submodule.dilation = (dilate, dilate)
            submodule.padding = (dilate, dilate)

def recover_dilate_module(module):
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.Conv2d) and (submodule.kernel_size == 3 or submodule.kernel_size == (3,3)):
            submodule.dilation = (1, 1)
            submodule.padding = (1, 1)

def make_dilate_model(model, enable_dilate=False, dilate=2, nskip=3):
    if not enable_dilate:
        recover_dilate_module(model.model.diffusion_model)
    else:
        nin = 0
        for inblock in model.model.diffusion_model.input_blocks:
            nin += 1
            if nskip >= nin:
                pass
            else:
                make_dilate_module(inblock, dilate)
                
        for midblock in model.model.diffusion_model.middle_block:
            make_dilate_module(midblock, dilate)

        nout = 0
        for outblock in model.model.diffusion_model.output_blocks:
            nout += 1
            if nskip > len(model.model.diffusion_model.output_blocks) - nout:
                pass
            else:
                make_dilate_module(outblock, dilate)
