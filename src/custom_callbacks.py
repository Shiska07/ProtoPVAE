from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

'''
Some components of the following implementation were obtained from: https://github.com/cfchen-duke/ProtoPNet
 '''

def getmodel_ckpt_callback(model_ckpt_path, filename="{epoch}-{val_loss:.3f}-{"
                                                     "val_acc:.4f}",
                           metric="epoch", mode="max"):
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,
        dirpath=model_ckpt_path,
        filename=filename,
        save_top_k=5,
        mode=mode,
        save_on_train_epoch_end=True,
        every_n_epochs=1,
        save_weights_only=True
    )
    return checkpoint_callback


