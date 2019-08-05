# pylint: disable=protected-access
import os
import glob
from typing import List, Optional

import torch


class CheckpointManager:
    def __init__(self, path: str, retain: int) -> None:
        """
        Constructor.

        Args:
            path (str): The path to the checkpoint directory.
            retain (int): Specifies the maximum number of checkpoint files that can
            be saved.

        Returns:
            None
        """
        self.path = path
        self.retain = retain

    def make_checkpoint(self, module: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        epoch: int, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> None:
        """
        Makes a checkpoint and saves the necessary details.

        Args:
            module (torch.nn.Module): The module to be checkpointed.
            optimizer (torch.optim.Optimizer): The optimizer.
            epoch (int): The current epoch.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler.

        Returns:
            None
        """

        files = self.get_files_in_dir(module.name)
        
        if len(files) >= self.retain:
            for f in files[self.retain - 1:]:
                os.remove(f)  # remove file

        checkpoint_name = module.name + "_epoch_" + str(epoch) + ".pth.tar"
        save_dict = {
            'state_dict': module.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if scheduler is not None:
            save_dict["scheduler"] = scheduler.state_dict()

        torch.save(save_dict, self.path + "/" + checkpoint_name)
        
    def restore(self, module: torch.nn.Module, optimizer: torch.optim.Optimizer,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> int:
        """
        Restores the module from the latest checkpoint.
        
        Args:
            module (torch.nn.Module): The module to restore.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler.

        Returns:
            int: The epoch to resume from.
        """

        files = self.get_files_in_dir(module.name)
        if not files:
            return False

        checkpoint_file = files[0]
        # loading checkpoint data
        data = torch.load(checkpoint_file)

        # loading data to objects
        module.load_state_dict(data['state_dict'])
        optimizer.load_state_dict(data['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(data['scheduler'])

        return data['epoch']

    def get_files_in_dir(self, module_name: str) -> List[str]:
        """
        Gets the files of the module in the checkpoint directory and sorts them according to
        date modified, in descending order.

        Args:
            module_name: The name of the module whose files are required.

        Returns:
            List[str]: The list of file paths.
        """
        # getting contents of checkpoint dir
        files = glob.glob(self.path + "/" + module_name + "*")
        # storing only the files
        files = [f for f in files if os.path.isfile(f)]
        # sorting the files according to modification time
        files.sort(key=os.path.getmtime, reverse=True)

        return files
