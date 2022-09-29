"""Verify that a de novo sequencing model will run."""
import pytorch_lightning as pl
from depthcharge.models import Spec2Pep
from depthcharge.data import AnnotatedSpectrumIndex, SpectrumDataModule


def test_denovo_model(tmp_path, mgf_small):
    """Test de novo model overfitting"""
    pl.seed_everything(42)
    index = AnnotatedSpectrumIndex(tmp_path / "test.hdf5", mgf_small)
    loaders = SpectrumDataModule(
        index, num_workers=0, batch_size=2, shuffle=False
    )
    loaders.setup()

    # Create the model
    model = Spec2Pep(16, n_layers=1, n_log=20, dim_intensity=4)
    trainer = pl.Trainer(
        logger=None,
        max_epochs=200,
        enable_checkpointing=False,
    )

    # Fit the model
    trainer.fit(model, loaders.train_dataloader())

    # Predict
    pred = trainer.predict(model, loaders.train_dataloader())
    assert pred[0][0][0][0] == "$LESLLEK"
