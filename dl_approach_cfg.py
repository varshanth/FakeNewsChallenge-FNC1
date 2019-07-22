TRAIN_PATH = 'fnc_pkl/train_datapoints.pkl'
TEST_PATH = 'fnc_pkl/test_datapoints.pkl'

LEN_HEADLINE = 15
LEN_BODY = 60

TRAIN_CFG = {
         'BATCH_SIZE' : 32,
         'N_EPOCHS' : 5,
         'WEIGHTS_PATH' : 'model_chkpts/cond_cnn_classif',
         'PATIENCE' : 1,
         'LR' : 0.001,
         'LR_DECAY_STEPS' : 10,
         'LR_DECAY_GAMMA' : 0.1,
         }

DATA_CFG = {
        'MAX_VOCAB_SIZE' : 40000,
        'VECTORS': 'glove.6B.100d',
        'VECTOR_SIZE' : 100,
        }

VANILLA_COND_CNN_NET_CFG = {
        'h_num_filt' : 256,
        'h_n_list' : [2,3,4],
        'b_num_filt' : 192,
        'b_n_list' : [2,3,4,5],
        'num_classes' : None, # To fill dynamically
        'dropout_rate' : 0.4,
        }

POS_TAGGED_COND_CNN_NET_CFG = {
        'h_num_filt' : 256,
        'h_n_list' : [1,2,3],
        'b_num_filt' : 256,
        'b_n_list' : [1,2,3],
        'num_classes' : None, # To fill dynamically
        'dropout_rate' : 0.4,
        }

SHARED_CONV_VANILLA_COND_CNN_NET_CFG = {
        'num_filt' : 256,
        'n_list' : [2,3,4],
        'num_classes' : None, # To fill dynamically
        'dropout_rate' : 0.4,
        }

SHARED_CONV_POS_TAGGED_COND_CNN_NET_CFG = {
        'num_filt' : 256,
        'n_list' : [1,2,3],
        'num_classes' : None, # To fill dynamically
        'dropout_rate' : 0.4,
        }

EMBED_CFG = {
        'H_V' : None, # To fill dynamically
        'B_V' : None, # To fill dynamically
        'D' : DATA_CFG['VECTOR_SIZE']
        }


