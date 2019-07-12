TRAIN_PATH = {
        'stances': 'fnc-1/train_stances.csv',
        'bodies' : 'fnc-1/train_bodies.csv'
        }

TEST_PATH = {
        'stances' : 'fnc-1/competition_test_stances.csv',
        'bodies'  : 'fnc-1/competition_test_bodies.csv'
        }

LEN_HEADLINE = 15
LEN_BODY = 60

TRAIN_CFG = {
         'BATCH_SIZE' : 32,
         'N_EPOCHS' : 15,
         'WEIGHTS_PATH' : 'model_chkpts/cond_cnn_classif',
         'PATIENCE' : 4,
         'LR' : 0.001,
         'LR_DECAY_STEPS' : 10,
         'LR_DECAY_GAMMA' : 0.1,
         }

DATA_CFG = {
        'MAX_VOCAB_SIZE' : 40000,
        'VECTORS': 'glove.6B.100d',
        'VECTOR_SIZE' : 100,
        }

NET_CFG = {
        'h_num_filt' : 256,
        'h_n_list' : [2,3,4],
        'b_num_filt' : 192,
        'b_n_list' : [2,3,4,5],
        'num_h_tokens' : LEN_HEADLINE,
        'num_classes' : None, # To fill dynamically
        'dropout_rate' : 0.4,
        }

EMBED_CFG = {
        'H_V' : None, # To fill dynamically
        'B_V' : None, # To fill dynamically
        'D' : DATA_CFG['VECTOR_SIZE']
        }


