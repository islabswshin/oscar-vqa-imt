from datetime import datetime
import os
import platform
import wandb


def time_log() -> str:
    a = datetime.now()
    s = "-" * 72
    s += f"  {a.year:>4}/{a.month:>2}/{a.day:>2} | {a.hour:>2}:{a.minute:>2}:{a.second:>2}"
    return s


def wandb_setup(cfg: dict) -> str:
    run_type = cfg["run_type"]
    save_dir = cfg["save_dir"] # root save dir

    wandb_mode = cfg["wandb"]["mode"].lower()
    if wandb_mode not in ("online", "offline", "disabled"):
        raise ValueError(f"[ERROR:WANDB] Mode {wandb_mode} invalid.")

    os.makedirs(save_dir, exist_ok=True)

    wandb_project = cfg["project"]
    wandb_name = cfg["name"]

    wandb_note = cfg["wandb"]["notes"] if "notes" in cfg["wandb"] else None
    wandb_id = cfg["wandb"]["id"] if "id" in cfg["wandb"] else None
    server_name = platform.node()
    wandb_note = server_name + (f"-{wandb_note}" if (wandb_note is not None) else "")

    wandb.init(project=wandb_project,
               job_type=run_type,
               name=wandb_name,
               dir=save_dir,
               resume="allow",
               mode=wandb_mode,
               notes=wandb_note,
               config=cfg,
               id=wandb_id)
    save_path = wandb.run.dir if (wandb_mode != "disabled") else save_dir
    return save_path