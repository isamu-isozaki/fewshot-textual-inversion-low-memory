import argparse
from glob import glob
import os
import torch
torch.cuda.empty_cache()
import numpy as np
import torch as th
th.autograd.set_detect_anomaly(True)
import torchvision.transforms as T
from tqdm import trange

from glide_finetune.glide_finetune import run_glide_finetune_epoch
from glide_finetune.glide_util import load_model, add_tokens_and_get_placeholder_token
from glide_finetune.loader import TextualInversionDataset
from glide_finetune.train_util import wandb_setup
from glide_finetune.wds_loader import glide_wds_loader
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import set_seed

def run_glide_finetune(
    args,
    placeholder_token="<cat_toy>",
    initializer_token="toy",
    data_dir="./data",
    batch_size=1,
    learning_rate=1e-5,
    adam_weight_decay=0.0,
    side_x=64,
    side_y=64,
    resize_ratio=1.0,
    uncond_p=0.0,
    resume_ckpt="",
    checkpoints_dir="./finetune_checkpoints",
    use_fp16=False,
    device="cpu",
    freeze_transformer=True,
    freeze_diffusion=True,
    project_name="glide_finetune",
    activation_checkpointing=False,
    use_captions=True,
    num_epochs=100,
    log_frequency=100,
    sample_bs=1,
    sample_gs=8.0,
    use_webdataset=False,
    image_key="jpg",
    caption_key="txt",
    enable_upsample=False,
    upsample_factor=4,
    image_to_upsample='low_res_face.png',
    inpainting=False,
    repeats=100,
    learnable_property='object',
    center_crop=False,
    scale_lr=True,
    adam_beta1=0.9,
    adam_beta2=0.999,
):
    if "~" in data_dir:
        data_dir = os.path.expanduser(data_dir)
    if "~" in checkpoints_dir:
        checkpoints_dir = os.path.expanduser(checkpoints_dir)

    # Create the checkpoint/output directories
    os.makedirs(checkpoints_dir, exist_ok=True)
    if scale_lr:
        learning_rate = batch_size*learning_rate
    # Start wandb logging
    wandb_run = wandb_setup(args, project_name)
    print("Wandb setup.")

    # Model setup
    model_type = "base" if not enable_upsample else "upsample"
    if inpainting:
        model_type = f"{model_type}-inpaint"
    glide_model, glide_diffusion, glide_options = load_model(
        glide_path=resume_ckpt,
        use_fp16=use_fp16,
        freeze_transformer=freeze_transformer,
        freeze_diffusion=freeze_diffusion,
        activation_checkpointing=activation_checkpointing,
        model_type=model_type,
    )
    tokenizer = glide_model.tokenizer
    token_ids = tokenizer.encode(initializer_token)
    print(f"number of tokens: {len(token_ids)}")
    placeholder_token, placeholder_token_ids = add_tokens_and_get_placeholder_token(args, token_ids, tokenizer, glide_model)
    if args.subject_noun:
        placeholder_token = f"{placeholder_token} {args.subject_noun}"
    glide_model.train()
    number_of_params = sum(x.numel() for x in glide_model.parameters())
    glide_model.token_embedding.requires_grad_(True)

    print(f"Number of parameters: {number_of_params}")
    print('Parameters that requires gradient:')
    number_of_trainable_params = sum(
        x.numel() for x in glide_model.parameters() if x.requires_grad
    )
    for name, param in glide_model.named_parameters():
        if param.requires_grad:
            print(name)
    print(f"Trainable parameters: {number_of_trainable_params}")

    # Data setup
    print("Loading data...")
    if use_webdataset:
        dataset = glide_wds_loader(
            urls=data_dir,
            caption_key=caption_key,
            image_key=image_key,
            enable_image=True,
            enable_text=use_captions,
            enable_upsample=enable_upsample,
            tokenizer=glide_model.tokenizer,
            ar_lower=0.5,
            ar_upper=2.0,
            min_original_height=side_x * upsample_factor,
            min_original_width=side_y * upsample_factor,
            upscale_factor=upsample_factor,
            nsfw_filter=True,
            similarity_threshold_upper=0.0,
            similarity_threshold_lower=0.5,
            words_to_skip=[],
            dataset_name="laion",  # can be laion, alamy.
        )
    else:
        dataset = TextualInversionDataset(
            data_root=data_dir,
            placeholder_token=placeholder_token,
            repeats=repeats,
            learnable_property=learnable_property,
            center_crop=center_crop,
            set="train",
            side_x=side_x,
            side_y=side_y,
            resize_ratio=resize_ratio,
            uncond_p=uncond_p,
            tokenizer=glide_model.tokenizer,
            text_ctx_len=glide_options["text_ctx"],
            enable_glide_upsample=enable_upsample,
            upscale_factor=upsample_factor,  # TODO: make this a parameter
        )

    # Data loader setup
    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not use_webdataset,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # Optimizer setup
    eps = 1e-8
    if args.use_fp16:
        eps = 1e-4
    optimizer = th.optim.AdamW(
        glide_model.token_embedding.parameters(),
        lr=learning_rate,
        weight_decay=0,
        eps=eps
    )


    # Training setup
    outputs_dir = "./outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    existing_runs = [ sub_dir for sub_dir in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, sub_dir))]
    existing_runs_int = []
    for x in existing_runs:
        try:
            existing_runs_int.append(int(x))
        except:
            print("unexpected directory naming scheme")
            #ignore
    existing_runs_int = sorted(existing_runs_int)
    next_run = 0 if len(existing_runs) == 0 else existing_runs_int[-1] + 1
    current_run_ckpt_dir = os.path.join(checkpoints_dir, str(next_run).zfill(4))

    os.makedirs(current_run_ckpt_dir, exist_ok=True)
    prompt = f"A picture of {placeholder_token}"
    if learnable_property != "object":
        prompt = f"A painting in a style of {placeholder_token}"

    for epoch in trange(num_epochs):
        print(f"Starting epoch {epoch}")
        run_glide_finetune_epoch(
            args,
            placeholder_token_ids,
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            glide_options=glide_options,
            optimizer=optimizer,
            dataloader=dataloader,
            prompt=prompt,
            sample_bs=sample_bs,
            sample_gs=sample_gs,
            checkpoints_dir=current_run_ckpt_dir,
            outputs_dir=outputs_dir,
            side_x=side_x,
            side_y=side_y,
            device=device,
            wandb_run=wandb_run,
            log_frequency=log_frequency,
            epoch=epoch,
            gradient_accumualation_steps=1,
            train_upsample=enable_upsample,
            inpainting=inpainting,
            learning_rate=learning_rate
        )



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject_noun",
        type=str,
        default=None,
        help=(
            "Inspired by dream booth. Make the model guess the identifier in the form [identifier] subject noun"
        ),
    )
    parser.add_argument(
        "--num_vec_per_token",
        type=int,
        default=1,
        help=(
            "The number of vectors used to represent the placeholder token. The higher the number, the better the"
            " result at the cost of editability. This can be fixed by prompt editing."
        ),
    )
    parser.add_argument(
        "--guess_initializer_token",
        action="store_true",
        help="Guess the string the represent the concept using blip.",
    )
    parser.add_argument(
        "--initialize_rest_random",
        action="store_true",
        help="Initialize rest of the placeholder tokens with random.",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    parser.add_argument(
        "--learnable_property", type=str, default="object",help="object of style"
    )
    parser.add_argument("--scale_lr", action="store_true")

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--data_dir", "-data", type=str, default="./data")
    parser.add_argument("--data_aug",  action="store_true")

    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4)
    parser.add_argument("--side_x", "-x", type=int, default=64)
    parser.add_argument("--side_y", "-y", type=int, default=64)
    parser.add_argument(
        "--resize_ratio", "-crop", type=float, default=0.8, help="Crop ratio"
    )
    parser.add_argument(
        "--uncond_p",
        "-p",
        type=float,
        default=0.2,
        help="Probability of using the empty/unconditional token instead of a caption. OpenAI used 0.2 for their finetune.",
    )
    parser.add_argument(
        "--train_upsample",
        "-upsample",
        default=False,
        action="store_true",
        help="Train the upsampling type of the model instead of the base model.",
    )
    parser.add_argument(
        "--resume_ckpt",
        "-resume",
        type=str,
        default="",
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--checkpoints_dir", "-ckpt", type=str, default="./glide_checkpoints/"
    )
    parser.add_argument("--use_fp16", "-fp16", action="store_true")
    parser.add_argument("--device", "-dev", type=str, default="cuda")
    parser.add_argument("--log_frequency", "-freq", type=int, default=100)
    parser.add_argument("--project_name", "-name", type=str, default="glide-finetune")
    parser.add_argument("--activation_checkpointing", "-grad_ckpt", action="store_true")
    parser.add_argument("--use_captions", "-txt", action="store_true")
    parser.add_argument("--epochs", "-epochs", type=int, default=20)
    parser.add_argument(
        "--test_batch_size",
        "-tbs",
        type=int,
        default=1,
        help="Batch size used for model eval, not training.",
    )
    parser.add_argument(
        "--test_guidance_scale",
        "-tgs",
        type=float,
        default=4.0,
        help="Guidance scale used during model eval, not training.",
    )
    parser.add_argument(
        "--use_webdataset",
        "-wds",
        action="store_true",
        help="Enables webdataset (tar) loading",
    )
    parser.add_argument(
        "--wds_image_key",
        "-wds_img",
        type=str,
        default="jpg",
        help="A 'key' e.g. 'jpg' used to access the image in the webdataset",
    )
    parser.add_argument(
        "--wds_caption_key",
        "-wds_cap",
        type=str,
        default="txt",
        help="A 'key' e.g. 'txt' used to access the caption in the webdataset",
    )
    parser.add_argument(
        "--wds_dataset_name",
        "-wds_name",
        type=str,
        default="laion",
        help="Name of the webdataset to use (laion or alamy)",
    )
    parser.add_argument("--seed", "-seed", type=int, default=0)
    parser.add_argument(
        "--cudnn_benchmark",
        "-cudnn",
        action="store_true",
        help="Enable cudnn benchmarking. May improve performance. (may not)",
    )
    parser.add_argument(
        "--upscale_factor", "-upscale", type=int, default=4, help="Upscale factor for training the upsampling model only"
    )
    parser.add_argument("--image_to_upsample", "-lowres", type=str, default="low_res_face.png")
    parser.add_argument("--inpainting",  action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # CUDA/CPU setup
    args = parse_args()
    args.freeze_transformer=True
    args.freeze_diffusion=True

    if len(args.device) > 0:
        device = th.device(args.device)
    else:
        device = th.device("cpu") if not th.cuda.is_available() else th.device("cuda")

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    th.backends.cudnn.benchmark = args.cudnn_benchmark

    for arg in vars(args):
        print(f"--{arg} {getattr(args, arg)}")

    if args.use_webdataset:
        # webdataset uses tars
        data_dir = glob(os.path.join(args.data_dir, "*.tar"))
    else:
        data_dir = args.data_dir
    
    run_glide_finetune(
        args,
        placeholder_token=args.placeholder_token,
        initializer_token=args.initializer_token,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        adam_weight_decay=args.adam_weight_decay,
        side_x=args.side_x,
        side_y=args.side_y,
        resize_ratio=args.resize_ratio,
        uncond_p=args.uncond_p,
        resume_ckpt=args.resume_ckpt,
        checkpoints_dir=args.checkpoints_dir,
        use_fp16=args.use_fp16,
        device=device,
        log_frequency=args.log_frequency,
        freeze_transformer=args.freeze_transformer,
        freeze_diffusion=args.freeze_diffusion,
        project_name=args.project_name,
        activation_checkpointing=args.activation_checkpointing,
        use_captions=args.use_captions,
        num_epochs=args.epochs,
        sample_bs=args.test_batch_size,
        sample_gs=args.test_guidance_scale,
        use_webdataset=args.use_webdataset,
        image_key=args.wds_image_key,
        caption_key=args.wds_caption_key,
        enable_upsample=args.train_upsample,
        upsample_factor=args.upscale_factor,
        image_to_upsample=args.image_to_upsample,
        inpainting=args.inpainting,
        scale_lr=args.scale_lr,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
    )
