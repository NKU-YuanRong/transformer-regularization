from vit_pytorch import ViT

def get_model(args):
    return ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = 10,
        dim = 128,
        depth = 6,
        heads = 8,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1,
        channels=1
    )