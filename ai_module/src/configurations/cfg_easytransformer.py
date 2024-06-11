class easyAttn_config:
    """
    A class of configuration of Transformer predictor 
    """ 
    
    
    in_dim      = 8
    out_dim     = 8 # The output sequence length
    d_model     = 8 #(embedding size?)

    time_proj   = 8 # The projection on time, which is used for new embedding stragtegy 
    next_step   = 1
    nmode       = 16  # Should be consistent as the modes 
    

    num_head    = 4
    attn_type   = "easy" 
    embed       = "time" # sin / cos/ time
    num_block   = 4      # Number of layer 

    is_res_attn = True
    is_res_proj = True
    proj_dim    = 128

    act_proj    = "relu"
    is_output   = True
    out_act     = None

    Epoch       = 100
    Batch_size  = 256
    lr          = 1e-3

    train_split = 0.8 
    num_train   = 135000

    early_stop  = True

    if early_stop == True:
        patience  = 50 # 30 or 50
    else:
        patience  = 0 

