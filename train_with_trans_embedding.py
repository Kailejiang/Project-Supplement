import os
import schnetpack as spk
import schnetpack.transform as trn
import torch
import torchmetrics
import pytorch_lightning as pl

# Please modify all paths and parameters in the program according to your own research requirements.

# model saving path
save_path = './model_save'
if not os.path.exists('model_save'):
    os.makedirs(save_path)

# model loading
custom_data = spk.data.AtomsDataModule(
    './your_database_name.db',
    batch_size=20,
    distance_unit='Ang',
    property_units={"energy_U0":'eV'},
    num_train=1600,
    num_val=200,
    num_test=200,
    transforms=[
        trn.ASENeighborList(cutoff=6.),
        trn.RemoveOffsets("energy_U0", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    num_workers=1,
    pin_memory=True, # set to false, when not using a GPU
)
custom_data.prepare_data()

if __name__ == '__main__':
    custom_data.setup()

    # hyperparameter setting
    cutoff = 6.
    n_atom_basis = 128
    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=40, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=6,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

    # choose whether to use the embedded layer
    # please select "load_embedding = False" for initial training
    # please ensure that the model hyperparameters are consistent
    load_embedding = False
    if load_embedding:
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, 'embedding_weights.pth')
        if os.path.isfile(file_path):
            loaded_weights = torch.load('embedding_weights.pth')
            schnet.load_embedding_weights(loaded_weights)

    pred_U0 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="energy_U0")

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_U0],
        postprocessors=[trn.CastTo64(), trn.AddOffsets("energy_U0", add_mean=True, add_atomrefs=False)]
    )

    output_U0 = spk.task.ModelOutput(
        name="energy_U0",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_U0],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-5}
    )

    # logs saving
    logger = pl.loggers.CSVLogger(save_dir='logs', name='Log_1')
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(save_path, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=save_path,
        max_epochs=100,
    )
    trainer.fit(task, datamodule=custom_data)

    # trained embedding layer saving
    embedding_weights = schnet.get_embedding_weights()
    torch.save(embedding_weights, 'embedding_weights.pth')