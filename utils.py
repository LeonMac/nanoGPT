import os

A100_FLPS = 312e12
RTX3090_FLPS = 312e12

def obtain_filename():
    """Return the filename of the current script."""
    return os.path.basename(__file__)

def checkout_para_by_GPU(gpu:str):
    if gpu == 'A100':
        # A100 8 x GPU
        # these make the total batch size be ~0.5M
        # 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
        batch_size = 12
        block_size = 1024
        gradient_accumulation_steps = 5 * 8
        flps = A100_FLPS
    elif gpu == 'RTX3090':
        # 3090 1 x GPU
        # these make the total batch size be 
        # 18 batch size * 1024 block size * 5 gradaccum * 2 GPUs = 
        batch_size = 16 
        block_size = 1024
        gradient_accumulation_steps = 5  
        flps = RTX3090_FLPS
    elif gpu == 'dual-RTX3090':
        # 3090 2 x GPU
        # these make the total batch size be 
        # 18 batch size * 1024 block size * 5 gradaccum * 2 GPUs = 
        batch_size = 20 
        block_size = 1024
        gradient_accumulation_steps = 5 * 2
        flps = RTX3090_FLPS

    print(f"By using GPU {gpu} overide the batch_size = {batch_size},\
    block_size = {block_size}, gradient_accumulation_steps = {gradient_accumulation_steps}, flps = {flps}")

    return batch_size, block_size, gradient_accumulation_steps, flps

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
