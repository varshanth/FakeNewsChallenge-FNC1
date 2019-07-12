import torch


class EarlyStoppingWithSaveWeights(object):
    def __init__(self,
                 model,
                 patience,
                 save_weights_path, # Weights_Directory/Weights_File_Prefix
                 loss_band_low_frac = 0.8):
        self.patience = patience
        self.best_loss = float('inf')
        self.model = model
        self.save_weights_path = save_weights_path
        self.bad_epochs = 0
        self.models_saved = 0
        self.loss_band_low_frac = loss_band_low_frac

    def step(self, curr_epoch_loss):
        if curr_epoch_loss < self.best_loss:
            self.bad_epochs = 0
            self.best_loss = curr_epoch_loss
            self.models_saved += 1
            save_weights_path = '{0}_{1}.pth'.format(
                    self.save_weights_path, self.models_saved)
            print('Saving Weights to: {}'.format(save_weights_path))
            torch.save(self.model.state_dict(),
                        save_weights_path)

        elif curr_epoch_loss >= (self.loss_band_low_frac * self.best_loss):
            self.bad_epochs += 1
            print('Bad epoch {}'.format(self.bad_epochs))

        else:
            self.bad_epochs = 0
        if self.bad_epochs > self.patience:
            print('Early Stopping Activated: Patience Level Surpassed')
            return True
        else:
            return False

