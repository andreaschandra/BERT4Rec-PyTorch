from .bert import BERTTrainer


TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    return BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)
