import os

A100_FLPS = 312e12
RTX3090_FLPS = 312e12

def obtain_filename():
    """Return the filename of the current script."""
    return os.path.basename(__file__)

# def checkout_para_by_GPU(gpu:str, init_from:str = 'gpt'):
#     if gpu == 'A100':
#         # A100 8 x GPU
#         # these make the total batch size be ~0.5M
#         # 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
#         batch_size = 12
#         block_size = 1024
#         gradient_accumulation_steps = 5 * 8
#         flps = A100_FLPS
#     elif gpu == 'RTX3090':
#         # 3090 1 x GPU
#         # these make the total batch size be 
#         # 18 batch size * 1024 block size * 5 gradaccum * 2 GPUs = 
#         batch_size = 16 
#         block_size = 224 #1024 gpt, 512 medium, 224 large, 64 xl  
#         gradient_accumulation_steps = 5  
#         flps = RTX3090_FLPS
#     elif gpu == 'dual-RTX3090':
#         # 3090 2 x GPU
#         # these make the total batch size be 
#         # 18 batch size * 1024 block size * 5 gradaccum * 2 GPUs = 
#         batch_size = 18
#         block_size = 1024
#         gradient_accumulation_steps = 5 * 2
#         flps = RTX3090_FLPS

#     print(f"By using GPU {gpu} overide the batch_size = {batch_size},\
#     block_size = {block_size}, gradient_accumulation_steps = {gradient_accumulation_steps}, flps = {flps}")

#     return batch_size, block_size, gradient_accumulation_steps, flps

def s_to_hhmmss(seconds: int):
    # Calculate hours, minutes, and seconds
    days  = seconds // 86400        
    hours = (seconds // 3600) %  24  
    seconds %= 3600
    minutes = seconds // 60    
    seconds %= 60
    seconds = int(seconds)     

    # Format the time as "hh:mm:ss"
    return f"d:h:m:s [{days:02}][{hours:02}]:[{minutes:02}]:[{seconds:02}]"

gpt_dict = {
    "gpt2": {
        "A100": {
            "batch_size": 12,
            "block_size": 1024,
            "gradient_accumulation_steps": 40,
            "flps": A100_FLPS
        },
        "RTX3090": {
            "batch_size": 16,
            "block_size": 1024,
            "gradient_accumulation_steps": 5,
            "flps": RTX3090_FLPS
        },
        "dual-RTX3090": {
            "batch_size": 16,
            "block_size": 1024,
            "gradient_accumulation_steps": 10,
            "flps": RTX3090_FLPS
        }
    },
    "gpt2-medium": {
        "A100": {
            "batch_size": 12,
            "block_size": 512,
            "gradient_accumulation_steps": 40,
            "flps": A100_FLPS
        },
        "RTX3090": {
            "batch_size": 16,
            "block_size": 512,
            "gradient_accumulation_steps": 5,
            "flps": RTX3090_FLPS
        },
        "dual-RTX3090": {
            "batch_size": 16,
            "block_size": 512,
            "gradient_accumulation_steps": 10,
            "flps": RTX3090_FLPS
        }
    },
    "gpt2-large": {
        "A100": {
            "batch_size": 12,
            "block_size": 256,
            "gradient_accumulation_steps": 40,
            "flps": A100_FLPS
        },
        "RTX3090": {
            "batch_size": 16,
            "block_size": 224,
            "gradient_accumulation_steps": 5,
            "flps": RTX3090_FLPS
        },
        "dual-RTX3090": {
            "batch_size": 16,
            "block_size": 224,
            "gradient_accumulation_steps": 10,
            "flps": RTX3090_FLPS
        }
    },
    "gpt2-xl": {
        "A100": {
            "batch_size": 12,
            "block_size": 16,
            "gradient_accumulation_steps": 40,
            "flps": A100_FLPS
        },
        "RTX3090": {
            "batch_size": 1,
            "block_size": 4,
            "gradient_accumulation_steps": 5,
            "flps": RTX3090_FLPS
        },
        "dual-RTX3090": {
            "batch_size": 1,
            "block_size": 4,
            "gradient_accumulation_steps": 10,
            "flps": RTX3090_FLPS
        }
    },
    "other": {
        "A100": {
            "batch_size": 12,
            "block_size": 1024,
            "gradient_accumulation_steps": 8,
            "flps": A100_FLPS
        },
        "RTX3090": {
            "batch_size": 16,
            "block_size": 1024,
            "gradient_accumulation_steps": 5,
            "flps": RTX3090_FLPS
        },
        "dual-RTX3090": {
            "batch_size": 16,
            "block_size": 1024,
            "gradient_accumulation_steps": 10,
            "flps": RTX3090_FLPS
        }
    } 
}

def checkout_para_by_GPU(device:str, model:str = 'gpt'):
    model = model if model in gpt_dict else 'other'
    try:
        batch_size = gpt_dict[model][device]['batch_size']
        block_size = gpt_dict[model][device]['block_size']
        gradient_accumulation_steps = gpt_dict[model][device]['gradient_accumulation_steps']
        flps = gpt_dict[model][device]['flps']

        print(f"By using GPU {device} and model init from {model}, overide the batch_size = {batch_size}, block_size = {block_size}, gradient_accumulation_steps = {gradient_accumulation_steps}, flps = {flps}")

        return batch_size, block_size, gradient_accumulation_steps, flps

    except KeyError as e:
        print(f"Error: {e} not found in the dictionary.")
        return None, None, None, None